# IBM Research Singapore, 2022

# taken from https://gist.github.com/circulosmeos/ef6497fd3344c2c2508b92bb9831173f 
import hashlib
import base58

# ECDSA bitcoin Public Key
# ubkey = m.groups()[0]

def pubkey2hash160(
    pubkey, 
    compress_pubkey = False,
):
# See 'compressed form' at https://en.bitcoin.it/wiki/Protocol_documentation#Signatures

    def hash160(hex_str):
        sha = hashlib.sha256()
        rip = hashlib.new('ripemd160')
        sha.update(hex_str)
        rip.update( sha.digest() )
        # print ( "key_hash = \t" + rip.hexdigest() )
        return rip.hexdigest()  # .hexdigest() is hex ASCII


    if (compress_pubkey):
        if (ord(bytearray.fromhex(pubkey[-2:])) % 2 == 0):
            pubkey_compressed = '02'
        else:
            pubkey_compressed = '03'
        pubkey_compressed += pubkey[2:66]
        hex_str = bytearray.fromhex(pubkey_compressed)
    else:
        hex_str = bytearray.fromhex(pubkey)

    # Obtain key:

    key_hash = '00' + hash160(hex_str)

    sha = hashlib.sha256()
    sha.update( bytearray.fromhex(key_hash) )
    checksum = sha.digest()
    sha = hashlib.sha256()
    sha.update(checksum)
    checksum = sha.hexdigest()[0:8]

    # print ( "checksum = \t" + sha.hexdigest() )
    # print ( "key_hash + checksum = \t" + key_hash + ' ' + checksum )
    # print ( "bitcoin address = \t" + (base58.b58encode( bytes(bytearray.fromhex(key_hash + checksum)) )).decode('utf-8') )
    return base58.b58encode( bytes(bytearray.fromhex(key_hash + checksum)) ).decode('utf-8') 

import pandas as pd
from functools import reduce
from typing import List, Optional, Any

def merge_dfs(
    dfs: List[pd.DataFrame],
    on : List[str] = ['txid'],
    how: str = 'outer',
    fillna: Optional[Any] = None,
):
    dfs = [df.set_index(on) for df in dfs]
    df = reduce(lambda left,right: left.join(right, how=how), dfs)
    if fillna is not None:
        df.fillna(fillna, inplace=True)
    return df.reset_index()


