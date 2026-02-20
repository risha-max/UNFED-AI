// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "../src/UnfedToken.sol";
import "../src/UnfedEscrow.sol";

contract UnfedEscrowTest is Test {
    UnfedToken token;
    UnfedEscrow escrow;

    address operator = address(this);
    address nodeA = address(0xA);
    address nodeB = address(0xB);
    address client = address(0xC);

    uint256 constant MIN_STAKE = 100 ether;
    uint256 constant COOLDOWN = 300;       // 5 minutes
    uint256 constant CHALLENGE = 60;       // 1 minute
    uint256 constant SLASH_BPS = 5000;     // 50%

    function setUp() public {
        token = new UnfedToken(1_000_000 ether);
        escrow = new UnfedEscrow(
            address(token),
            MIN_STAKE,
            COOLDOWN,
            CHALLENGE,
            UnfedEscrow.SlashDestination.Escrow,
            SLASH_BPS
        );

        // Distribute tokens to test accounts
        token.transfer(nodeA, 10_000 ether);
        token.transfer(nodeB, 10_000 ether);
        token.transfer(client, 50_000 ether);
    }

    // ---------------------------------------------------------------
    // Staking
    // ---------------------------------------------------------------

    function test_stake() public {
        vm.startPrank(nodeA);
        token.approve(address(escrow), 200 ether);
        escrow.stake(200 ether);
        vm.stopPrank();

        assertEq(escrow.stakedBalance(nodeA), 200 ether);
        assertTrue(escrow.isEligible(nodeA));
    }

    function test_stake_below_minimum_not_eligible() public {
        vm.startPrank(nodeA);
        token.approve(address(escrow), 50 ether);
        escrow.stake(50 ether);
        vm.stopPrank();

        assertEq(escrow.stakedBalance(nodeA), 50 ether);
        assertFalse(escrow.isEligible(nodeA));
    }

    function test_stake_zero_reverts() public {
        vm.startPrank(nodeA);
        vm.expectRevert("UnfedEscrow: zero amount");
        escrow.stake(0);
        vm.stopPrank();
    }

    // ---------------------------------------------------------------
    // Unstaking and withdrawal
    // ---------------------------------------------------------------

    function test_unstake_and_withdraw() public {
        // Stake first
        vm.startPrank(nodeA);
        token.approve(address(escrow), 200 ether);
        escrow.stake(200 ether);

        // Request unstake
        escrow.requestUnstake(200 ether);
        assertEq(escrow.stakedBalance(nodeA), 0);

        // Cannot withdraw before cooldown
        vm.expectRevert("UnfedEscrow: cooldown not expired");
        escrow.withdraw();

        // Warp past cooldown
        vm.warp(block.timestamp + COOLDOWN);

        uint256 balBefore = token.balanceOf(nodeA);
        escrow.withdraw();
        uint256 balAfter = token.balanceOf(nodeA);

        assertEq(balAfter - balBefore, 200 ether);
        vm.stopPrank();
    }

    function test_unstake_insufficient_reverts() public {
        vm.startPrank(nodeA);
        token.approve(address(escrow), 100 ether);
        escrow.stake(100 ether);

        vm.expectRevert("UnfedEscrow: insufficient stake");
        escrow.requestUnstake(200 ether);
        vm.stopPrank();
    }

    // ---------------------------------------------------------------
    // Client deposits
    // ---------------------------------------------------------------

    function test_client_deposit_and_withdraw() public {
        vm.startPrank(client);
        token.approve(address(escrow), 1000 ether);
        escrow.deposit(1000 ether);

        assertEq(escrow.clientBalances(client), 1000 ether);
        assertEq(escrow.escrowPool(), 1000 ether);

        // Withdraw half
        escrow.clientWithdraw(500 ether);
        assertEq(escrow.clientBalances(client), 500 ether);
        assertEq(escrow.escrowPool(), 500 ether);
        vm.stopPrank();
    }

    function test_client_withdraw_insufficient_reverts() public {
        vm.startPrank(client);
        token.approve(address(escrow), 100 ether);
        escrow.deposit(100 ether);

        vm.expectRevert("UnfedEscrow: insufficient balance");
        escrow.clientWithdraw(200 ether);
        vm.stopPrank();
    }

    // ---------------------------------------------------------------
    // Settlements
    // ---------------------------------------------------------------

    function test_settlement_full_flow() public {
        // Setup: client deposits, nodes stake
        vm.prank(client);
        token.approve(address(escrow), 1000 ether);
        vm.prank(client);
        escrow.deposit(1000 ether);

        vm.prank(nodeA);
        token.approve(address(escrow), 200 ether);
        vm.prank(nodeA);
        escrow.stake(200 ether);

        vm.prank(nodeB);
        token.approve(address(escrow), 200 ether);
        vm.prank(nodeB);
        escrow.stake(200 ether);

        // Post settlement (operator)
        bytes32 hash = keccak256("settlement-1");
        address[] memory nodes = new address[](2);
        nodes[0] = nodeA;
        nodes[1] = nodeB;
        uint256[] memory amounts = new uint256[](2);
        amounts[0] = 30 ether;
        amounts[1] = 20 ether;

        escrow.postSettlement(hash, nodes, amounts);
        assertEq(escrow.escrowPool(), 950 ether); // 1000 - 50

        // Cannot finalize during challenge window
        vm.expectRevert("UnfedEscrow: challenge window open");
        escrow.finalizeSettlement(hash);

        // Warp past challenge window
        vm.warp(block.timestamp + CHALLENGE);
        escrow.finalizeSettlement(hash);

        // Nodes can claim earnings
        assertEq(escrow.earnings(nodeA), 30 ether);
        assertEq(escrow.earnings(nodeB), 20 ether);

        vm.prank(nodeA);
        escrow.claimEarnings();
        assertEq(token.balanceOf(nodeA), 9830 ether); // 10000 - 200 (staked) + 30

        vm.prank(nodeB);
        escrow.claimEarnings();
        assertEq(token.balanceOf(nodeB), 9820 ether); // 10000 - 200 (staked) + 20
    }

    function test_settlement_duplicate_reverts() public {
        vm.prank(client);
        token.approve(address(escrow), 1000 ether);
        vm.prank(client);
        escrow.deposit(1000 ether);

        bytes32 hash = keccak256("dup");
        address[] memory nodes = new address[](1);
        nodes[0] = nodeA;
        uint256[] memory amounts = new uint256[](1);
        amounts[0] = 10 ether;

        escrow.postSettlement(hash, nodes, amounts);

        vm.expectRevert("UnfedEscrow: duplicate settlement");
        escrow.postSettlement(hash, nodes, amounts);
    }

    function test_settlement_insufficient_escrow_reverts() public {
        // No client deposits
        bytes32 hash = keccak256("no-funds");
        address[] memory nodes = new address[](1);
        nodes[0] = nodeA;
        uint256[] memory amounts = new uint256[](1);
        amounts[0] = 10 ether;

        vm.expectRevert("UnfedEscrow: insufficient escrow pool");
        escrow.postSettlement(hash, nodes, amounts);
    }

    // ---------------------------------------------------------------
    // Challenge settlement
    // ---------------------------------------------------------------

    function test_challenge_settlement() public {
        // Setup
        vm.prank(client);
        token.approve(address(escrow), 1000 ether);
        vm.prank(client);
        escrow.deposit(1000 ether);

        bytes32 hash = keccak256("fraudulent");
        address[] memory nodes = new address[](1);
        nodes[0] = nodeA;
        uint256[] memory amounts = new uint256[](1);
        amounts[0] = 50 ether;

        escrow.postSettlement(hash, nodes, amounts);
        assertEq(escrow.escrowPool(), 950 ether);

        // Challenge within window
        escrow.challengeSettlement(hash);

        // Funds returned to escrow
        assertEq(escrow.escrowPool(), 1000 ether);

        // Cannot finalize a challenged settlement
        vm.warp(block.timestamp + CHALLENGE);
        vm.expectRevert("UnfedEscrow: settlement was challenged");
        escrow.finalizeSettlement(hash);
    }

    function test_challenge_after_window_reverts() public {
        vm.prank(client);
        token.approve(address(escrow), 1000 ether);
        vm.prank(client);
        escrow.deposit(1000 ether);

        bytes32 hash = keccak256("late-challenge");
        address[] memory nodes = new address[](1);
        nodes[0] = nodeA;
        uint256[] memory amounts = new uint256[](1);
        amounts[0] = 10 ether;

        escrow.postSettlement(hash, nodes, amounts);

        // Warp past window
        vm.warp(block.timestamp + CHALLENGE);

        vm.expectRevert("UnfedEscrow: window expired");
        escrow.challengeSettlement(hash);
    }

    // ---------------------------------------------------------------
    // Slashing
    // ---------------------------------------------------------------

    function test_slash_node() public {
        // Node stakes
        vm.prank(nodeA);
        token.approve(address(escrow), 200 ether);
        vm.prank(nodeA);
        escrow.stake(200 ether);

        // Fund escrow for checking slash destination
        vm.prank(client);
        token.approve(address(escrow), 100 ether);
        vm.prank(client);
        escrow.deposit(100 ether);

        uint256 escrowBefore = escrow.escrowPool();

        // Operator slashes (50% of 200 = 100)
        escrow.slashNode(nodeA);

        assertEq(escrow.stakedBalance(nodeA), 100 ether); // 200 - 100
        // Slash destination is Escrow, so escrow pool grows
        assertEq(escrow.escrowPool(), escrowBefore + 100 ether);
    }

    function test_slash_during_cooldown() public {
        // Node stakes then requests unstake
        vm.prank(nodeA);
        token.approve(address(escrow), 200 ether);
        vm.prank(nodeA);
        escrow.stake(200 ether);

        vm.prank(nodeA);
        escrow.requestUnstake(200 ether);

        // Active stake is 0, but 200 in cooldown
        assertEq(escrow.stakedBalance(nodeA), 0);

        // Operator can still slash (50% of 200 cooldown = 100)
        escrow.slashNode(nodeA);

        // After slash, cooldown amount reduced
        (uint256 amt, ) = escrow.unstakeRequests(nodeA);
        assertEq(amt, 100 ether);
    }

    function test_slash_nothing_reverts() public {
        vm.expectRevert("UnfedEscrow: nothing to slash");
        escrow.slashNode(nodeA);
    }

    function test_slash_non_operator_reverts() public {
        vm.prank(nodeA);
        token.approve(address(escrow), 200 ether);
        vm.prank(nodeA);
        escrow.stake(200 ether);

        vm.prank(nodeB);
        vm.expectRevert("UnfedEscrow: caller is not the operator");
        escrow.slashNode(nodeA);
    }

    // ---------------------------------------------------------------
    // Config updates
    // ---------------------------------------------------------------

    function test_update_config() public {
        escrow.setMinStake(200 ether);
        assertEq(escrow.minStake(), 200 ether);

        escrow.setCooldownPeriod(600);
        assertEq(escrow.cooldownPeriod(), 600);

        escrow.setChallengeWindow(120);
        assertEq(escrow.challengeWindow(), 120);

        escrow.setSlashFractionBps(7500);
        assertEq(escrow.slashFractionBps(), 7500);

        escrow.setSlashDestination(UnfedEscrow.SlashDestination.Burn);
        assertEq(uint(escrow.slashDestination()), uint(UnfedEscrow.SlashDestination.Burn));
    }

    function test_config_non_operator_reverts() public {
        vm.prank(nodeA);
        vm.expectRevert("UnfedEscrow: caller is not the operator");
        escrow.setMinStake(0);
    }

    function test_transfer_operator() public {
        escrow.transferOperator(nodeA);
        assertEq(escrow.operator(), nodeA);

        // Old operator can no longer call operator functions
        vm.expectRevert("UnfedEscrow: caller is not the operator");
        escrow.setMinStake(0);
    }
}
