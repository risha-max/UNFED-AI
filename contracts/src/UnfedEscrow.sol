// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/// @title UnfedEscrow — On-chain staking, escrow, and settlement for UNFED AI.
///
/// @notice Each cluster deploys one instance of this contract, parameterised
///         by an ERC-20 token chosen by the cluster operator.  The contract
///         handles:
///
///   1. **Node staking** — Nodes lock tokens to prove commitment.  The
///      registry reads `stakedBalance()` to gate registration.
///   2. **Client deposits** — Clients pre-fund an escrow balance that pays
///      for inference.
///   3. **Batch settlements** — The operator posts periodic settlement
///      batches (from the share-chain).  Each settlement has a challenge
///      window; after it expires the payouts are released to nodes.
///   4. **Slashing** — The operator can slash a node's stake (operator-
///      attested fraud).  Slashed funds are routed to burn, operator, or
///      back to the client escrow pool (configurable).
///
/// @dev Trust model: the operator is trusted for slashing and settlement
///      posting, analogous to a mining pool operator.  Nodes and clients
///      accept this by choosing the cluster.  Market competition between
///      clusters keeps operators honest.
contract UnfedEscrow {
    using SafeERC20 for IERC20;

    // ---------------------------------------------------------------
    // Types
    // ---------------------------------------------------------------

    enum SlashDestination {
        Burn,      // 0 — tokens sent to address(0xdead)
        Operator,  // 1 — tokens go to operator
        Escrow     // 2 — tokens returned to client escrow pool
    }

    struct UnstakeRequest {
        uint256 amount;
        uint256 readyAt;  // block.timestamp after which withdrawal is allowed
    }

    struct Settlement {
        address[] nodes;
        uint256[] amounts;
        uint256 totalPayout;
        uint256 postedAt;
        uint256 challengeDeadline;
        bool finalized;
        bool challenged;
    }

    // ---------------------------------------------------------------
    // State
    // ---------------------------------------------------------------

    IERC20 public immutable token;
    address public operator;

    // Config (updatable by operator)
    uint256 public minStake;
    uint256 public cooldownPeriod;
    uint256 public challengeWindow;
    SlashDestination public slashDestination;
    uint256 public slashFractionBps; // basis points (e.g. 5000 = 50%)

    // Node staking
    mapping(address => uint256) public stakes;
    mapping(address => UnstakeRequest) public unstakeRequests;

    // Client escrow
    mapping(address => uint256) public clientBalances;
    uint256 public escrowPool; // aggregate client funds available for settlements

    // Settlements
    mapping(bytes32 => Settlement) public settlements;

    // Node earnings (claimable balance)
    mapping(address => uint256) public earnings;

    // ---------------------------------------------------------------
    // Events
    // ---------------------------------------------------------------

    event Staked(address indexed node, uint256 amount);
    event UnstakeRequested(address indexed node, uint256 amount, uint256 readyAt);
    event Withdrawn(address indexed node, uint256 amount);
    event Deposited(address indexed client, uint256 amount);
    event ClientWithdrawn(address indexed client, uint256 amount);
    event SettlementPosted(bytes32 indexed hash, uint256 totalPayout, uint256 deadline);
    event SettlementFinalized(bytes32 indexed hash);
    event SettlementChallenged(bytes32 indexed hash);
    event Slashed(address indexed node, uint256 amount, SlashDestination destination);
    event EarningsClaimed(address indexed node, uint256 amount);
    event ConfigUpdated(string key);

    // ---------------------------------------------------------------
    // Modifiers
    // ---------------------------------------------------------------

    modifier onlyOperator() {
        require(msg.sender == operator, "UnfedEscrow: caller is not the operator");
        _;
    }

    // ---------------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------------

    /// @param _token           ERC-20 token used for staking and payments
    /// @param _minStake        Minimum staked balance for node eligibility
    /// @param _cooldownPeriod  Seconds a node must wait after requesting unstake
    /// @param _challengeWindow Seconds a settlement can be challenged
    /// @param _slashDest       Where slashed funds go (0=burn, 1=operator, 2=escrow)
    /// @param _slashBps        Slash percentage in basis points (5000 = 50%)
    constructor(
        address _token,
        uint256 _minStake,
        uint256 _cooldownPeriod,
        uint256 _challengeWindow,
        SlashDestination _slashDest,
        uint256 _slashBps
    ) {
        require(_token != address(0), "UnfedEscrow: zero token address");
        require(_slashBps <= 10000, "UnfedEscrow: slash > 100%");

        token = IERC20(_token);
        operator = msg.sender;
        minStake = _minStake;
        cooldownPeriod = _cooldownPeriod;
        challengeWindow = _challengeWindow;
        slashDestination = _slashDest;
        slashFractionBps = _slashBps;
    }

    // ---------------------------------------------------------------
    // Node Staking
    // ---------------------------------------------------------------

    /// @notice Lock tokens as stake.  Caller must have approved this contract.
    function stake(uint256 amount) external {
        require(amount > 0, "UnfedEscrow: zero amount");
        token.safeTransferFrom(msg.sender, address(this), amount);
        stakes[msg.sender] += amount;
        emit Staked(msg.sender, amount);
    }

    /// @notice Begin unstaking.  Funds remain slashable during cooldown.
    function requestUnstake(uint256 amount) external {
        require(amount > 0, "UnfedEscrow: zero amount");
        require(stakes[msg.sender] >= amount, "UnfedEscrow: insufficient stake");

        stakes[msg.sender] -= amount;
        unstakeRequests[msg.sender] = UnstakeRequest({
            amount: unstakeRequests[msg.sender].amount + amount,
            readyAt: block.timestamp + cooldownPeriod
        });
        emit UnstakeRequested(msg.sender, amount, block.timestamp + cooldownPeriod);
    }

    /// @notice Withdraw unstaked funds after cooldown.
    function withdraw() external {
        UnstakeRequest storage req = unstakeRequests[msg.sender];
        require(req.amount > 0, "UnfedEscrow: no pending unstake");
        require(block.timestamp >= req.readyAt, "UnfedEscrow: cooldown not expired");

        uint256 amount = req.amount;
        req.amount = 0;
        req.readyAt = 0;
        token.safeTransfer(msg.sender, amount);
        emit Withdrawn(msg.sender, amount);
    }

    /// @notice Check a node's active staked balance.
    function stakedBalance(address node) external view returns (uint256) {
        return stakes[node];
    }

    /// @notice Check if a node meets the minimum stake requirement.
    function isEligible(address node) external view returns (bool) {
        return stakes[node] >= minStake;
    }

    // ---------------------------------------------------------------
    // Client Deposits
    // ---------------------------------------------------------------

    /// @notice Deposit tokens into the client escrow pool.
    function deposit(uint256 amount) external {
        require(amount > 0, "UnfedEscrow: zero amount");
        token.safeTransferFrom(msg.sender, address(this), amount);
        clientBalances[msg.sender] += amount;
        escrowPool += amount;
        emit Deposited(msg.sender, amount);
    }

    /// @notice Withdraw unused client credits.
    function clientWithdraw(uint256 amount) external {
        require(clientBalances[msg.sender] >= amount, "UnfedEscrow: insufficient balance");
        clientBalances[msg.sender] -= amount;
        escrowPool -= amount;
        token.safeTransfer(msg.sender, amount);
        emit ClientWithdrawn(msg.sender, amount);
    }

    // ---------------------------------------------------------------
    // Settlements (operator only)
    // ---------------------------------------------------------------

    /// @notice Post a batch settlement from the share-chain.
    /// @param hash        Unique settlement hash (from SettlementSummary)
    /// @param nodes       Node addresses that earned payouts
    /// @param amounts     Payout amount per node (same order as nodes)
    function postSettlement(
        bytes32 hash,
        address[] calldata nodes,
        uint256[] calldata amounts
    ) external onlyOperator {
        require(nodes.length == amounts.length, "UnfedEscrow: length mismatch");
        require(settlements[hash].postedAt == 0, "UnfedEscrow: duplicate settlement");

        uint256 total = 0;
        for (uint256 i = 0; i < amounts.length; i++) {
            total += amounts[i];
        }
        require(escrowPool >= total, "UnfedEscrow: insufficient escrow pool");

        escrowPool -= total;

        settlements[hash] = Settlement({
            nodes: nodes,
            amounts: amounts,
            totalPayout: total,
            postedAt: block.timestamp,
            challengeDeadline: block.timestamp + challengeWindow,
            finalized: false,
            challenged: false
        });

        emit SettlementPosted(hash, total, block.timestamp + challengeWindow);
    }

    /// @notice Finalize a settlement after its challenge window.
    function finalizeSettlement(bytes32 hash) external {
        Settlement storage s = settlements[hash];
        require(s.postedAt > 0, "UnfedEscrow: unknown settlement");
        require(!s.finalized, "UnfedEscrow: already finalized");
        require(!s.challenged, "UnfedEscrow: settlement was challenged");
        require(block.timestamp >= s.challengeDeadline, "UnfedEscrow: challenge window open");

        s.finalized = true;

        for (uint256 i = 0; i < s.nodes.length; i++) {
            earnings[s.nodes[i]] += s.amounts[i];
        }

        emit SettlementFinalized(hash);
    }

    /// @notice Challenge a settlement (operator only — fraud detected off-chain).
    function challengeSettlement(bytes32 hash) external onlyOperator {
        Settlement storage s = settlements[hash];
        require(s.postedAt > 0, "UnfedEscrow: unknown settlement");
        require(!s.finalized, "UnfedEscrow: already finalized");
        require(block.timestamp < s.challengeDeadline, "UnfedEscrow: window expired");

        s.challenged = true;

        // Return funds to escrow pool (clients are made whole)
        escrowPool += s.totalPayout;

        emit SettlementChallenged(hash);
    }

    /// @notice Claim accumulated earnings.
    function claimEarnings() external {
        uint256 amount = earnings[msg.sender];
        require(amount > 0, "UnfedEscrow: no earnings");
        earnings[msg.sender] = 0;
        token.safeTransfer(msg.sender, amount);
        emit EarningsClaimed(msg.sender, amount);
    }

    // ---------------------------------------------------------------
    // Slashing (operator only)
    // ---------------------------------------------------------------

    /// @notice Slash a node's stake.  Operator-attested — the contract
    ///         trusts the operator's judgement.  The slashed amount is
    ///         routed according to `slashDestination`.
    /// @param node The node to slash
    function slashNode(address node) external onlyOperator {
        uint256 staked = stakes[node];

        // Also consider funds in cooldown (prevent cheat-then-unstake)
        uint256 inCooldown = unstakeRequests[node].amount;
        uint256 totalSlashable = staked + inCooldown;
        require(totalSlashable > 0, "UnfedEscrow: nothing to slash");

        uint256 slashAmount = (totalSlashable * slashFractionBps) / 10000;

        // Deduct from active stake first, then from cooldown
        if (slashAmount <= staked) {
            stakes[node] -= slashAmount;
        } else {
            uint256 remaining = slashAmount - staked;
            stakes[node] = 0;
            unstakeRequests[node].amount -= remaining;
        }

        // Route slashed funds
        if (slashDestination == SlashDestination.Burn) {
            // Send to dead address (effectively burned)
            token.safeTransfer(address(0xdead), slashAmount);
        } else if (slashDestination == SlashDestination.Operator) {
            token.safeTransfer(operator, slashAmount);
        } else {
            // SlashDestination.Escrow — return to client pool
            escrowPool += slashAmount;
        }

        emit Slashed(node, slashAmount, slashDestination);
    }

    // ---------------------------------------------------------------
    // Operator Config
    // ---------------------------------------------------------------

    function setMinStake(uint256 _minStake) external onlyOperator {
        minStake = _minStake;
        emit ConfigUpdated("minStake");
    }

    function setCooldownPeriod(uint256 _cooldownPeriod) external onlyOperator {
        cooldownPeriod = _cooldownPeriod;
        emit ConfigUpdated("cooldownPeriod");
    }

    function setChallengeWindow(uint256 _challengeWindow) external onlyOperator {
        challengeWindow = _challengeWindow;
        emit ConfigUpdated("challengeWindow");
    }

    function setSlashDestination(SlashDestination _dest) external onlyOperator {
        slashDestination = _dest;
        emit ConfigUpdated("slashDestination");
    }

    function setSlashFractionBps(uint256 _bps) external onlyOperator {
        require(_bps <= 10000, "UnfedEscrow: slash > 100%");
        slashFractionBps = _bps;
        emit ConfigUpdated("slashFractionBps");
    }

    function transferOperator(address newOperator) external onlyOperator {
        require(newOperator != address(0), "UnfedEscrow: zero address");
        operator = newOperator;
        emit ConfigUpdated("operator");
    }
}
