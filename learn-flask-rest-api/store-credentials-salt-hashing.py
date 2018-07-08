
"""
authentication: the user is the right person
authorisation: the user has execution rights for a specific WSGI app

Ways to secure a REST service?
1. HTTP Authorisation header for encrypted transmissions of credentials
2. with Secure Sockets Layer SSL: can use the HTTP Basic Authorization mode, 
    which includes a username and password
3. For more elaborate measures, we can use HTTP Digest Authorization, which
requires an exchange with the server to get a piece of data 
called a nonce that's used to create the digest in a more secure fashion.

The only thing that can be stored is a repeatable cryptographic hash of
password plus salt. Hash is one-way, cannot be easily decoded, esp with salt.

"""

# =============================================================================
# How salted password hashing works? 
# =============================================================================

import hashlib
hashlib.algorithms_available
"""
{'SHA', 'dsaWithSHA', 'md5', 'MD5', 'md4', 'shake_256', 
'DSA-SHA', 'sha', 'whirlpool', 'dsaEncryption', 'sha512', 'sha384', 'SHA384', 
'ripemd160', 'sha1', 'sha224', 'RIPEMD160', 'sha3_512', 'SHA1', 'MD4', 
'sha3_384', 'sha3_224', 'sha3_256', 'blake2s', 'sha256', 'ecdsa-with-SHA1', 
'SHA224', 'SHA256', 'shake_128', 'SHA512', 'DSA', 'blake2b'}
"""
from hashlib import sha256
import os

class Authentication:
    """ store username and only hashes of password """

    iterations= 1000
    __slots__ = ('username', 'salt', 'hash')

    def __init__( self, username, password ):
        """Works with bytes. Not Unicode strings. Attributes should be private. """
        self.username= username
        self.salt = os.urandom(24) # random 24 bytes string generated once and stored
        self.hash = self._iter_hash( self.iterations, self.salt, username, password )

    @staticmethod
    def _iter_hash( iterations, salt, username, password ):
        """ constant time hashing. Repeat iter times. """
        seed= salt + b":" + username + b":" + password # byte strings only
        for i in range(iterations):
            seed= sha256( seed ).digest()
        return seed

    def __eq__( self, other ):
        """ compare relevant attributes of instance """
        return self.username == other.username and self.hash == other.hash

    def __hash__( self, other ):
        """ instance of object becomes immutable once overriden """
        return hash(self.hash)

    def __repr__( self ):
        salt_x= "".join( "{0:x}".format(b) for b in self.salt )
        hash_x= "".join( "{0:x}".format(b) for b in self.hash )
        return "{username} {iterations:d}:{salt}:{hash}".format(
                username=self.username, 
                iterations=self.iterations,
                salt=salt_x, 
                hash=hash_x)

    def match( self, password ):
        """ Need matching username, salt, password and iteration count """
        test = self._iter_hash( self.iterations, self.salt, self.username, password )
        print(b"Test hash: " + test)
        print(b"Self.hash: " + self.hash)        
        return self.hash == test # Constant Time is Best


class Users( dict ):
    """ collection of users. Username as dict keys. """

    def __init__( self, *args, **kw ):
        super().__init__( *args, **kw )
        # Can never match -- keys are the same.
        self[""] = Authentication( b"__dummy__", b"Doesn't Matter" )

    def add( self, authentication ):
        if authentication.username == "": # prohibit resetting "" key
            raise KeyError( "Invalid Authentication" )
        self[authentication.username] = authentication

    def match( self, username, password ):
        if username in self and username != "":
            return self[username].match(password)
        else:
            return self[""].match("Definitely a wrong password.")


users = Users()
users.add( Authentication(b"Aladdin", b"open sesame") )
users.match(username=b"Aladdin", password=b"open sesame")
users.match(username=b"Aladdin", password=b"close sesame")


import base64

class Authenticate( WSGI ):
    """ WSGI application that checks the authentication header 
        in a request and updates the environment for validated users
        then calling the wrapping WSGI application (seperation of concerns) """
    def __init__( self, users, target_app ):
        self.users = users # a pool of users authentication
        self.target_app = target_app # another wrapped WSGI app

    def __call__( self, environ, start_response ):
        """ credentials must be in basic scheme and in base 64 encoding """
        if 'HTTP_AUTHORIZATION' in environ:
            scheme, credentials = environ['HTTP_AUTHORIZATION'].split()
            if scheme == "Basic":
                username, password = base64.b64decode( credentials ).split(b":")
                if self.users.match(username, password):
                    environ['Authenticate.username']= username # update environ with authenticated username
                    return self.target_app(environ, start_response) # invoke wrapped app

        status = '401 UNAUTHORIZED'
        headers = [('Content-type', 'text/plain; charset=utf-8'),
                   ('WWW-Authenticate', 'Basic realm="roulette@localhost"')]
        start_response(status, headers)
        return [ "Not authorized".encode('utf-8') ]


# =============================================================================
# 
# =============================================================================
iterations= 1000
salt= os.urandom(24) 
username = b'justin'
password = b'abcde'
b'abc' + 'abc'
a = _iter_hash( iterations, salt, username, password )
b = _iter_hash( iterations, os.urandom(24), username, password )
a == b
a = _iter_hash( iterations, salt, username, password )
b = _iter_hash( iterations, salt, username, b"wrong" )
a == b

x=sha256(b'abc')
x.digest?
x.digest_size?
x.hexdigest?
x.name?
x.update?














