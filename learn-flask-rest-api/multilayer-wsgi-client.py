
"""
URI must not change. Represents family of identities.
URI cannot be a state as states are dynamic.
</app/URI/>
/app/bet/ --> POST: create bets GET: return dict of bets placed
/app/wheel/ --> POST: spin table, compute payoffs

each URI has its own GET output and POST input JSON-encoded dict 

Application Methods
-- CRUD --
Create PUT
Retrieve GET
Update POST
Delete

"""
from collections import defaultdict
import random

import sys
import wsgiref.util
from wsgiref.simple_server import make_server
from wsgiref.validate import validator

import http.client
import json
import concurrent.futures
import time
from collections.abc import Callable


# =============================================================================
# WSGI Applcations to handle CRUD Methods and data processing
# =============================================================================
class WSGI( Callable ):
    """ WSGI interface for nesting calls """
    def __call__( self, environ, start_response ): # fix call inputs
        raise NotImplementedError

class RESTException( Exception ):
    """ Uncaught errors will return generic status 500 from wsgiref.
        Otherwise, RESTful errors 4xx """
    pass

class Wheel:
    """Abstract, zero bins omitted."""
    def __init__( self ):
        self.rng= random.Random()
        self.bins= [
            {str(n): (35,1),
            self.redblack(n): (1,1),
            self.hilo(n): (1,1),
            self.evenodd(n): (1,1),
            } for n in range(1,37)
        ]
    @staticmethod
    def redblack(n):
        return "Red" if n in (1, 3, 5, 7, 9, 12, 14, 16, 18,
                              19, 21, 23, 25, 27, 30, 32, 34, 36) else "Black"
    @staticmethod
    def hilo(n):
        return "Hi" if n >= 19 else "Lo"
    @staticmethod
    def evenodd(n):
        return "Even" if n % 2 == 0 else "Odd"
    def spin( self ):
        return self.rng.choice( self.bins )

class Zero:
    def __init__( self ):
        super().__init__()

class DoubleZero:
    def __init__( self ):
        super().__init__()
        self.bins += [ {'00': (35,1)} ]

class American( Zero, DoubleZero, Wheel ):
    """ American wheel for rank counting """
    pass

class Table:
    """ one player, multiple bets """
    def __init__( self, stake=100 ):
        self.bets = defaultdict(int) # {bet : amount}
        self.stake = stake

    def place_bet( self, name, amount ):
        self.bets[name] += amount

    def clear_bets( self, name ):
        self.bets= defaultdict(int)

    def resolve( self, spin ):
        """spin is a dict with bet:(x:y)."""
        details= []
        while self.bets:
            bet, amount= self.bets.popitem()
            if bet in spin:
                x, y = spin[bet]
                self.stake += amount*x/y
                details.append( (bet, amount, 'win') )
            else:
                self.stake -= amount
                details.append( (bet, amount, 'lose') )
        return details

class Roulette( WSGI ):
    def __init__( self, wheel ):
        self.table= Table(100)
        self.rounds= 0
        self.wheel= wheel

    def __call__( self, environ, start_response ):
        """ Multi layer REST Services, each nested app with their GET POST """
        #print( environ, file=sys.stderr )
        app= wsgiref.util.shift_path_info(environ) # 1. Parse.
        print("Roulette App requesting for URI: /{0}".format(request), 
              file=sys.stderr) # 2. Logging.

        try: # 3. Evaluate request. 
            if app.lower() == "player":
                return self.player_app( environ, start_response ) #4. Corresponding response.
            elif app.lower() == "bet":
                return self.bet_app( environ, start_response )
            elif app.lower() == "wheel":
                return self.wheel_app( environ, start_response )
            else:
                raise RESTException("404 NOT_FOUND",
                                    "Unknown app in {SCRIPT_NAME}/{PATH_INFO}".format_map(environ))

        except RESTException as e:
            status= e.args[0]
            headers = [('Content-type', 'text/plain; charset=utf-8')]
            start_response( status, headers, sys.exc_info() )
            return [ repr(e.args).encode("UTF-8") ]

    # =============================================================================
    # WSGI application that wraps local applications
    # =============================================================================

    def player_app( self, environ, start_response ):
        """ 
        GET: dict(stake, rounds)
        POST: None 
        """
        if environ['REQUEST_METHOD'] == 'GET':
            details= dict( stake= self.table.stake, rounds= self.rounds )
            status = '200 OK'
            headers = [('Content-type', 'application/json; charset=utf-8')]
            start_response(status, headers)
            return [ json.dumps( details ).encode('UTF-8') ] # serialise message 'details'
        
        else:
            # proper RESTful response on error 4xx
            raise RESTException("405 METHOD_NOT_ALLOWED",
                                "Method '{REQUEST_METHOD}' not allowed".format_map(environ))

    def bet_app( self, environ, start_response ):
        """ 
        GET: dict(player: current bets on table)
        POST: data stream attached to request. Place a list of bets. 
            Reponse after processing POST 
        """
        if environ['REQUEST_METHOD'] == 'GET':
            details = dict( self.table.bets )
            
        elif environ['REQUEST_METHOD'] == 'POST':
            size= int(environ['CONTENT_LENGTH']) # read and process input
            raw= environ['wsgi.input'].read(size).decode("UTF-8")
            """
            Browsers use the encoding implemented by 
            the urllib.parse.urlencode() module function.            
            
            In other cases, a separate encoding such as JSON is used to create 
            data structures that are easier to work with than the quoted data 
            produced by a web form.
            """
            try:
                data = json.loads( raw )
                if isinstance(data,dict): 
                    data= [data] # cast to list for for iter
                for detail in data:
                    self.table.place_bet( detail['bet'], int(detail['amount']) )
            except Exception as e:
                raise RESTException("403 FORBIDDEN", 
                                    "Bet {raw!r}".format(raw=raw))
            
            details = dict( self.table.bets )
                
        else:
            raise RESTException("405 METHOD_NOT_ALLOWED",
                                "Method '{REQUEST_METHOD}' not allowed".format_map(environ))

        status = '200 OK'
        headers = [('Content-type', 'application/json; charset=utf-8')]
        start_response(status, headers)
        return [ json.dumps(details).encode('UTF-8') ]

    def wheel_app( self, environ, start_response ):
        """
        GET: None
        POST: reject any data attached in request
        """
        if environ['REQUEST_METHOD'] == 'POST':

            size= environ['CONTENT_LENGTH']
            if size != '0': # validate there's a data stream attached in request
                raw= environ['wsgi.input'].read(int(size)) 
                # all data read then ignored. This prevents crashing when
                # closing socket with unread data. 
                raise RESTException("403 FORBIDDEN", 
                                    "Data '{raw!r}' not allowed".format(raw=raw))

            # run app to process message to respond
            spin= self.wheel.spin()
            payout = self.table.resolve( spin )
            self.rounds += 1
            details = dict(spin=spin, payout=payout, 
                           stake= self.table.stake, 
                           rounds= self.rounds )
            
            status = '200 OK'
            headers = [('Content-type', 'application/json; charset=utf-8')]
            start_response(status, headers)
            return [ json.dumps( details ).encode('UTF-8') ]
        
        else:
            raise RESTException("405 METHOD_NOT_ALLOWED",
                                "Method '{REQUEST_METHOD}' not allowed".format_map(environ))

# =============================================================================
# Server Side - WSGI Server to host WSGI APplications
# =============================================================================
def roulette_server(count=1):
    """
    This application validates the interface used by the roulette application;
    it decorates the various APIs with assert statements to provide some diagnostic
    information.
    """
    wheel= American()
    roulette= Roulette(wheel)
    debug= validator(roulette)
    httpd = make_server('', 8080, debug)
    if count is None:
        httpd.serve_forever()
    else:
        for c in range(count):
            httpd.handle_request()

# =============================================================================
# Client Side
# =============================================================================

def roulette_client(method="GET", path="/", data=None):
    """
    Generic client that works with variety of RESTful servers.
    Client makes 'GET' or 'POST' requests,
    encodes data attached to a POST request,
    decodes JSON documents it receives and returns its status code within 2xx

    rest.request(method, url, 
                 body=None, headers={}, *, encode_chunked=False)

    """
    rest= http.client.HTTPConnection('localhost', 8080)

    if data:
        header= {"Content-type": "application/json; charset=utf-8'"}
        params= json.dumps( data ).encode('UTF-8')
        rest.request(method, path, params, header) # attach serialised data
    else:
        rest.request(method, path)

    response= rest.getresponse()
    raw= response.read().decode("utf-8")

    if 200 <= response.status < 300:
        document= json.loads(raw)
        return document
    else:
        print( response.status, response.reason )
        print( response.getheaders() )
        print( raw )


# =============================================================================
# Simulate Get POST requests from CLIENT to SERVER 
# =============================================================================

with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.submit( roulette_server, 4 )
#    time.sleep(2) # Wait for the server to start
    print( roulette_client("GET", "/player/" ) )
    print( roulette_client("POST", "/bet/", {'bet':'Black', 'amount':2}) )
    print( roulette_client("GET", "/bet/" ) )
    print( roulette_client("POST", "/wheel/" ) )

# =============================================================================
"""
 print( environ, file=sys.stderr )
    wsgiref.util.shift_path_info?
        Shift a name from PATH_INFO to SCRIPT_NAME
"""
# =============================================================================
"""
{'ALLUSERSPROFILE': 'C:\\ProgramData', 
 'APPDATA': 'C:\\Users\\yeoshuiming\\AppData\\Roaming', 
 'ASL.LOG': 'Destination=file', 
 'COMMONPROGRAMFILES': 'C:\\Program Files\\Common Files', 
 'COMMONPROGRAMFILES(X86)': 'C:\\Program Files (x86)\\Common Files', 
 'COMMONPROGRAMW6432': 'C:\\Program Files\\Common Files', 
 'COMPUTERNAME': 'JYSM', 
 'COMSPEC': 'C:\\WINDOWS\\system32\\cmd.exe', 
 'CONDA_DEFAULT_ENV': 'base', 
 'CONDA_PREFIX': 'C:\\ProgramData\\Anaconda3', 
 'CONDA_PROMPT_MODIFIER': '(base) ', 
 'CONDA_PYTHON_EXE': 'C:\\ProgramData\\Anaconda3\\python.exe', 
 'CONDA_SHLVL': '1', 
 'DRIVERDATA': 'C:\\Windows\\System32\\Drivers\\DriverData', 
 'FPS_BROWSER_APP_PROFILE_STRING': 'Internet Explorer', 
 'FPS_BROWSER_USER_PROFILE_STRING': 'Default', 
 'FP_NO_HOST_CHECK': 'NO', 
 'FSHARPINSTALLDIR': 'C:\\Program Files (x86)\\Microsoft SDKs\\F#\\10.1\\Framework\\v4.0\\', 
 'HADOOP_HOME': 'C:\\spark-2.3.1-bin-hadoop2.7', 
 'HOMEDRIVE': 'C:', 
 'HOMEPATH': '\\Users\\yeoshuiming', 
 'JAVA_HOME': 'C:\\Java\\jdk-10.0.1', 
 'LOCALAPPDATA': 'C:\\Users\\yeoshuiming\\AppData\\Local', 
 'LOGONSERVER': '\\\\JYSM', 
 'MSMPI_BIN': 'C:\\Program Files\\Microsoft MPI\\Bin\\', 
 'NUMBER_OF_PROCESSORS': '8', 
 'ONEDRIVE': 'C:\\Users\\yeoshuiming\\OneDrive', 
 'OS': 'Windows_NT', 
 'PATH': 'C:\\ProgramData\\Anaconda3;C:\\ProgramData\\Anaconda3\\Library\\mingw-w64\\bin;C:\\ProgramData\\Anaconda3\\Library\\usr\\bin;C:\\ProgramData\\Anaconda3\\Library\\bin;C:\\ProgramData\\Anaconda3\\Scripts;C:\\ProgramData\\Anaconda3\\bin;C:\\Program Files (x86)\\Common Files\\Oracle\\Java\\javapath;C:\\Users\\yeoshuiming\\Anaconda3;C:\\Program Files\\Microsoft MPI\\Bin\\;C:\\Program Files (x86)\\Intel\\Intel(R) Management Engine Components\\iCLS\\;C:\\Program Files\\Intel\\Intel(R) Management Engine Components\\iCLS\\;C:\\WINDOWS\\system32;C:\\WINDOWS;C:\\WINDOWS\\System32\\Wbem;C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\;C:\\Program Files\\Intel\\WiFi\\bin\\;C:\\Program Files\\Common Files\\Intel\\WirelessCommon\\;C:\\Program Files (x86)\\Intel\\Intel(R) Management Engine Components\\DAL;C:\\Program Files\\Intel\\Intel(R) Management Engine Components\\DAL;C:\\Program Files (x86)\\Intel\\Intel(R) Management Engine Components\\IPT;C:\\Program Files\\Intel\\Intel(R) Management Engine Components\\IPT;C:\\Program Files\\Git\\cmd;C:\\Program Files\\nodejs\\;C:\\Program Files\\dotnet\\;C:\\Program Files\\Microsoft SQL Server\\130\\Tools\\Binn\\;C:\\Program Files (x86)\\Microsoft SQL Server\\140\\Tools\\Binn\\;C:\\Program Files\\Microsoft SQL Server\\140\\Tools\\Binn\\;C:\\Program Files (x86)\\Microsoft SQL Server\\140\\DTS\\Binn\\;C:\\Program Files\\Microsoft SQL Server\\140\\DTS\\Binn\\;C:\\Program Files\\Microsoft SQL Server\\Client SDK\\ODBC\\130\\Tools\\Binn\\;C:\\Program Files (x86)\\Microsoft SQL Server\\Client SDK\\ODBC\\130\\Tools\\Binn\\;C:\\Program Files (x86)\\Microsoft SQL Server\\140\\Tools\\Binn\\ManagementStudio\\;;C:\\WINDOWS\\System32\\OpenSSH\\;C:\\Program Files (x86)\\sbt\\bin;C:\\Program Files (x86)\\scala\\bin;C:\\Program Files\\Microsoft MPI\\Bin\\;C:\\Program Files (x86)\\Intel\\Intel(R) Management Engine Components\\iCLS\\;C:\\Program Files\\Intel\\Intel(R) Management Engine Components\\iCLS\\;C:\\WINDOWS\\system32;C:\\WINDOWS;C:\\WINDOWS\\System32\\Wbem;C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\;C:\\Program Files\\Intel\\WiFi\\bin\\;C:\\Program Files\\Common Files\\Intel\\WirelessCommon\\;C:\\Program Files (x86)\\Intel\\Intel(R) Management Engine Components\\DAL;C:\\Program Files\\Intel\\Intel(R) Management Engine Components\\DAL;C:\\Program Files (x86)\\Intel\\Intel(R) Management Engine Components\\IPT;C:\\Program Files\\Intel\\Intel(R) Management Engine Components\\IPT;C:\\Program Files\\Git\\cmd;C:\\Program Files\\nodejs\\;C:\\Program Files\\dotnet\\;C:\\Program Files\\Microsoft SQL Server\\130\\Tools\\Binn\\;C:\\Program Files (x86)\\Microsoft SQL Server\\140\\Tools\\Binn\\;C:\\Program Files\\Microsoft SQL Server\\140\\Tools\\Binn\\;C:\\Program Files (x86)\\Microsoft SQL Server\\140\\DTS\\Binn\\;C:\\Program Files\\Microsoft SQL Server\\140\\DTS\\Binn\\;C:\\Program Files\\Microsoft SQL Server\\Client SDK\\ODBC\\130\\Tools\\Binn\\;C:\\Program Files (x86)\\Microsoft SQL Server\\Client SDK\\ODBC\\130\\Tools\\Binn\\;C:\\Program Files (x86)\\Microsoft SQL Server\\140\\Tools\\Binn\\ManagementStudio\\;C:\\Users\\yeoshuiming\\Anaconda3;C:\\Users\\yeoshuiming\\Anaconda3\\Library\\mingw-w64\\bin;C:\\Users\\yeoshuiming\\Anaconda3\\Library\\usr\\bin;C:\\Users\\yeoshuiming\\Anaconda3\\Library\\bin;C:\\Users\\yeoshuiming\\Anaconda3\\Scripts;C:\\Users\\yeoshuiming\\AppData\\Local\\Microsoft\\WindowsApps;C:\\Users\\yeoshuiming\\AppData\\Roaming\\npm;C:\\Program Files\\Microsoft VS Code\\bin;R:=C:\\Program Files\\R\\R-3.5.0\\bin\\R.exe;C:\\Users\\yeoshuiming\\AppData\\Local\\Microsoft\\WindowsApps;C:\\Users\\yeoshuiming\\jython2.7.0\\bin;C:\\spark-2.3.1-bin-hadoop2.7\\bin;C:\\Java\\jdk-10.0.1\\bin;C:\\Program Files\\heroku\\bin;', 
 'PATHEXT': '.COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC', 
 'PROCESSOR_ARCHITECTURE': 'AMD64', 
 'PROCESSOR_IDENTIFIER': 'Intel64 Family 6 Model 142 Stepping 10, GenuineIntel', 
 'PROCESSOR_LEVEL': '6', 
 'PROCESSOR_REVISION': '8e0a', 
 'PROGRAMDATA': 'C:\\ProgramData', 
 'PROGRAMFILES': 'C:\\Program Files', 
 'PROGRAMFILES(X86)': 'C:\\Program Files (x86)', 
 'PROGRAMW6432': 'C:\\Program Files', 
 'PROMPT': '(base) $P$G', 
 'PSMODULEPATH': 'C:\\WINDOWS\\system32\\WindowsPowerShell\\v1.0\\Modules\\;C:\\Program Files (x86)\\Microsoft SQL Server\\140\\Tools\\PowerShell\\Modules\\', 
 'PUBLIC': 'C:\\Users\\Public', 
 'PYTHONIOENCODING': '1252', 
 'PYTHONPATH': 'C:\\Users\\yeoshuiming\\Dropbox\\GitHub\\py-learn\\python-oop', 
 'R_HOME': 'C:\\Program Files\\R\\R-3.5.0', 
 'SBT_HOME': 'C:\\Program Files (x86)\\sbt\\', 
 'SESSIONNAME': 'Console', 
 'SPARK_HOME': 'C:\\spark-2.3.1-bin-hadoop2.7', 
 'SYSTEMDRIVE': 'C:', 
 'SYSTEMROOT': 'C:\\WINDOWS', 
 'TEMP': 'C:\\Users\\YEOSHU~1\\AppData\\Local\\Temp', 
 'TMP': 'C:\\Users\\YEOSHU~1\\AppData\\Local\\Temp', 
 'USERDOMAIN': 'JYSM', 
 'USERDOMAIN_ROAMINGPROFILE': 'JYSM', 
 'USERNAME': 'yeoshuiming', 
 'USERPROFILE': 'C:\\Users\\yeoshuiming', 
 'WINDIR': 'C:\\WINDOWS', 
 
 'SERVER_NAME': 'JYSM', 
 'GATEWAY_INTERFACE': 'CGI/1.1', 
 'SERVER_PORT': '8080', 
 'REMOTE_HOST': '', 
 'CONTENT_LENGTH': '0', 
 'SCRIPT_NAME': '', 
 'SERVER_PROTOCOL': 'HTTP/1.1', 
 'SERVER_SOFTWARE': 'WSGIServer/0.2', 
 'REQUEST_METHOD': 'POST', 
 'PATH_INFO': '/wheel/', 
 'QUERY_STRING': '', 
 'REMOTE_ADDR': '127.0.0.1', 
 'CONTENT_TYPE': 'text/plain', 
 'HTTP_HOST': 'localhost:8080', 
 'HTTP_ACCEPT_ENCODING': 'identity', 
 'wsgi.input': <wsgiref.validate.InputWrapper object at 0x000001EA03AF6F98>, 
 'wsgi.errors': <wsgiref.validate.ErrorWrapper object at 0x000001EA03AF6D68>, 
 'wsgi.version': (1, 0), 
 'wsgi.run_once': False, 
 'wsgi.url_scheme': 'http', 
 'wsgi.multithread': True, 
 'wsgi.multiprocess': False, 
 'wsgi.file_wrapper': <class 'wsgiref.util.FileWrapper'>}
 """
 