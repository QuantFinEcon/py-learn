
import sys
import wsgiref.util
from wsgiref.simple_server import make_server
import json
from collections.abc import Callable

class RESTException( Exception ):
    """ Uncaught errors will return generic status 500 from wsgiref.
        Otherwise, RESTful errors 4xx """
    pass


# =============================================================================
# local apps to host in server
# =============================================================================
class app1:
    """ app to run on wsgi server.
        stateful configuration with bins
        (encapsulate data) """
    def __init__(self):
        print("Running app1")
        self.bins = [{self.method1(k): "Ran method (k)".format(k=k)} 
                        for k in range(5)]
    
    @staticmethod
    def method1(k):
        return "LessThanTwo" if k <2 else "MoreThanTwo"
    
    def run(self):
        return self.bins

class app2(app1, Callable):
    """ extend base stateful app1 to include WSGI interface.
        Make it a callable obj WSGI .
        Higher level wraper app delegate job to downstream.
        (encapsulate processing) """
    def __call__(self, environ, start_response):
        """ 
        APP handles GET POST        
        callable push wsgi processing: evaluation and response downstream
        """
        if environ['REQUEST_METHOD'] == 'GET':
            response = self.run() # 3. Evaluate.
            status = '200 OK' # 4. Respond.
            headers = [('Content-type', 'application/json; charset=utf-8')]
            start_response(status, headers)
            return [ json.dumps(response).encode('UTF-8') ]
        else:
            raise RESTException("405 METHOD_NOT_ALLOWED",
                                "Method '{REQUEST_METHOD}' not allowed".format_map(environ))
    
class default_app:
    def __init__(self):
        print("Running default_app")
        super().__init__() # inherit bins by mixin
        self.bins += [{'default':"Ran method default"}]

class transaction(default_app, app2):
    # MRO mixins trick to search for super().__init__()
    # inherit default_app first but init after app2
    # app2 is Callable
    pass

# Don't keep global variables, encapsulate data
transact = transaction()
transact.run()

# =============================================================================
# setup wsgi server to host local apps
# =============================================================================

def wsgi_app(environ, start_response):
    """ application to host on server """
    request= wsgiref.util.shift_path_info(environ) # 1. Parse.
    print( "wsgi_app requesting for URI: /{0}".format(request), file=sys.stderr ) # 2. Logging.

    try:
        if request.lower() == "transact": # 3. Evaluate.
            app_output = transact.run()
        else:
#            environ = dict(SCRIPT_NAME='a',PATH_INFO='b')
            raise RESTException("404 APP_NOT_FOUND", 
                                "Unknown app in {SCRIPT_NAME}/{PATH_INFO}".format_map(environ))
    except RESTException as e:
        status= e.args[0]
        headers = [('Content-type', 'text/plain; charset=utf-8')]
        start_response( status, headers, sys.exc_info() )
        return [ repr(e.args).encode("UTF-8") ]

    status = '200 OK' # 4. Respond.
    headers = [('Content-type', 'application/json; charset=utf-8')]
    start_response(status, headers)
    return [ json.dumps(app_output).encode('UTF-8') ]

class wsgi_app2(Callable):
    """ make wsgi_app from function to stateful callable obj.
        Place to tweak the nested app environment 
        in the wrapping application """
    def __init__(self):
        """ nested apps """
        self.transact = transaction() # __init__ callable app
        self.transact2 = transaction()

    def __call__(self, environ, start_response):
        """ 
        (envrion,start_response) args --> transaction subclass
            --> superclass callable object app2(envrion,start_response)
        e.g. higher order function f(g(x))
        """
        request = wsgiref.util.shift_path_info(environ) # 1. Parse.
        print( "wsgi_app2 requesting for URI: /{0}"\
              .format(request), file=sys.stderr ) # 2. Logging.
        try:
            # multi layered REST apps
            if request.lower()=='transact': # 3 + 4 wrapped downstream
                response= self.transact(environ,start_response)
            elif request.lower()=='transact2':
                response= self.transact2(environ,start_response)
            else:
                # not a valid request URI
                raise RESTException("404 APP_NOT_FOUND", 
                                    "Unknown app in {SCRIPT_NAME}/{PATH_INFO}".format_map(environ))
            
        except RESTException as e: # not encapsulated yet
            status= e.args[0]
            headers = [('Content-type', 'text/plain; charset=utf-8')]
            start_response( status, headers, sys.exc_info() )
            return [ repr(e.args).encode("UTF-8") ]
            
        return response # 4. Respond (Encapsulated in app2.__call__)

def wsgi_server(request_count=1):
    """ run this function from the command line in a terminal window """
    # invoke wsgi_app for each GET request
    httpd = make_server(host='localhost', port=8080, app=wsgi_app2)
    print("wsgi server started!")

    if request_count is None:
        print("wsgi listening... ")
        httpd.serve_forever()
    else:
        # in Unittest, fix number of requests to handle
        for c in range(request_count):
            httpd.handle_request()

# simple example of GET and server handle request
"""
httpd = make_server(host='localhost', port=8080, app=wsgi_app)

# need recreate rest HTTPConnection for each request
rest= http.client.HTTPConnection(host='localhost', port=8080) # 1. connect to server
rest.request("GET", "/transact/") # 2. send request
httpd.handle_request()
response = rest.getresponse()
response.getcode()
response.status
response.reason
response.getheaders()
response.geturl()
x=response.read()
x
x.decode("utf-8") # as string
json.loads(x) # as objects

dir(response)
'begin', 'chunk_left', 'chunked', 'close', 'closed', 'code', 
'debuglevel', 'detach', 'fileno', 'flush', 'fp', 'getcode', 
'getheader', 'getheaders', 'geturl', 'headers', 'info', 'isatty', 
'isclosed', 'length', 'msg', 'peek', 'read', 'read1', 'readable', 
'readinto', 'readinto1', 'readline', 'readlines', 'reason', 'seek', 
'seekable', 'status', 'tell', 'truncate', 'version', 
'will_close', 'writable', 'write', 'writelines'


httpd.application # app on hosting server
httpd.base_environ
httpd.server_address
httpd.socket_type
httpd.socket
httpd.server_close()
httpd.socket # closed
httpd.server_activate()

#httpd.get_request(request,client_address) # Get the request and client address from the socket.
#httpd.verify_request(request, client_address)
#httpd.finish_request(request, client_address) #same as process_request
#httpd.close_request(request)

dir(httpd)
'_handle_request_noblock', 'address_family', 'allow_reuse_address', 
'application', 'base_environ', 'close_request', 'fileno', 'finish_request', 
'get_app', 'get_request', 'handle_error', 'handle_request', 'handle_timeout', 
'process_request', 'request_queue_size', 'serve_forever', 
'server_activate', 'server_address', 'server_bind', 'server_close', 
'server_name', 'server_port', 'service_actions', 'set_app', 'setup_environ', 
'shutdown', 'shutdown_request', 'socket', 'socket_type', 'timeout', 'verify_request']

# possible errors

1. CannotSendRequest: Request-sent
    exception is raised when you reuse httplib.HTTP object 
    for new request while you havn't called 
    its getresponse() method for previous request

2. ConnectionRefusedError: [WinError 10061] No connection could be made 
                            because the target machine actively refused it
    server have close sockets, hence HTTPconnection request to server apps fail
    server not running in background i.e. run in seperate terminal
"""

# =============================================================================
# 
# =============================================================================
if __name__ == "__main__":
    wsgi_server(None) # serve forever

