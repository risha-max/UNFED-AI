// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "../src/UnfedToken.sol";
import "../src/UnfedEscrow.sol";

/// @notice Deploys UnfedToken (test ERC-20) and UnfedEscrow to a local
///         Anvil chain.  The deployer becomes the cluster operator.
///
/// Usage:
///   anvil --port 8545 &
///   forge script script/Deploy.s.sol --rpc-url http://localhost:8545 --broadcast
contract DeployScript is Script {
    function run() external {
        // Anvil's first default private key
        uint256 deployerKey = vm.envOr(
            "PRIVATE_KEY",
            uint256(0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80)
        );

        vm.startBroadcast(deployerKey);

        // 1. Deploy test token (1 million UNFED, 18 decimals)
        UnfedToken token = new UnfedToken(1_000_000 ether);

        // 2. Deploy escrow contract
        //    - minStake:        100 UNFED
        //    - cooldownPeriod:  300s  (5 min for dev, would be hours in prod)
        //    - challengeWindow: 60s   (1 min for dev, would be hours in prod)
        //    - slashDest:       Escrow (slashed funds return to client pool)
        //    - slashBps:        5000  (50%)
        UnfedEscrow escrow = new UnfedEscrow(
            address(token),
            100 ether,                       // minStake
            300,                             // cooldownPeriod (seconds)
            60,                              // challengeWindow (seconds)
            UnfedEscrow.SlashDestination.Escrow,
            5000                             // slashFractionBps (50%)
        );

        vm.stopBroadcast();

        // Log deployed addresses (captured by the shell script)
        console.log("TOKEN_ADDRESS=%s", address(token));
        console.log("ESCROW_ADDRESS=%s", address(escrow));
    }
}
