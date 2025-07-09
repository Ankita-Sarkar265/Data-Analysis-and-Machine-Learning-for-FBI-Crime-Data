# blockchain.py
import hashlib
import json
from datetime import datetime

class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        """
        Create the first block of the blockchain with fixed values.
        """
        genesis_block = self.create_block(data="Genesis Block", previous_hash="0")
        self.chain.append(genesis_block)

    def create_block(self, data, previous_hash):
        """
        Create a new block with given data and the hash of the previous block.
        """
        block = {
            "index": len(self.chain),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": data,
            "previous_hash": previous_hash,
        }
        block["hash"] = self.hash_block(block)
        return block

    def add_block(self, data):
        """
        Add a new block to the blockchain with the given data.
        """
        last_block = self.chain[-1]
        new_block = self.create_block(data, last_block["hash"])
        self.chain.append(new_block)
        return new_block

    def get_full_chain(self):
        """
        Return the entire blockchain.
        """
        return self.chain

    def verify_integrity(self):
        """
        Verify the blockchain by checking the hashes of each block.
        Returns True if the chain is valid, otherwise False.
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            # Check previous hash consistency
            if current_block["previous_hash"] != previous_block["hash"]:
                print(f"[ERROR] Block {i} has invalid previous hash!")
                return False

            # Recalculate the current block's hash and compare
            recalculated_hash = self.hash_block({
                "index": current_block["index"],
                "timestamp": current_block["timestamp"],
                "data": current_block["data"],
                "previous_hash": current_block["previous_hash"],
            })
            if current_block["hash"] != recalculated_hash:
                print(f"[ERROR] Block {i} hash mismatch!")
                return False

        return True

    @staticmethod
    def hash_block(block):
        """
        Calculate the SHA-256 hash of a block (excluding the hash itself).
        """
        block_copy = block.copy()
        block_copy.pop("hash", None)
        block_string = json.dumps(block_copy, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
