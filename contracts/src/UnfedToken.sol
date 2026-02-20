// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

/// @title UnfedToken — Test ERC-20 for local development.
/// @notice Mints an initial supply to the deployer.  In production, the
///         cluster operator would configure any ERC-20 (ETH wrapper, USDC,
///         custom token, etc.) — this is just the dev stand-in.
contract UnfedToken is ERC20 {
    constructor(uint256 initialSupply) ERC20("Unfed Token", "UNFED") {
        _mint(msg.sender, initialSupply);
    }
}
