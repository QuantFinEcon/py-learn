
import sys
import wsgiref.util
from wsgiref.simple_server import make_server
import json

# =============================================================================
# local apps to host in server
# =============================================================================
class app1:
    """ app to run on wsgi server """
    def __init__(self):
        print("Running app1")
        self.bins = [{self.method1(k): "Ran method (k)".format(k=k)} 
                        for k in range(5)]
    
    @staticmethod
    def method1(k):
        return "LessThanTwo" if k <2 else "MoreThanTwo"
    
    def run(self):
        return self.bins

class app2:
    def __init__(self):
        print("Running app2")
        super().__init__() # inherit bins
        self.bins += [{'default':"Ran method default"}]

class transaction(app2, app1):
    # MRO trick to search for super().__init__()
    # inherit app2 first but init after app1
    pass

transact = transaction()
transact.run()

# =============================================================================
# setup wsgi server to host local apps
# =============================================================================
class RESTException(Exception):
    pass

def wsgi_app(environ, start_response):

    request= wsgiref.util.shift_path_info(environ) # 1. Parse.
    print( "wsgi_app requesting for URI: /{0}".format(request), file=sys.stderr ) # 2. Logging.

    try:
        if request.lower() == "transact": # 3. Evaluate.
            app_output = transact.run()
        else:
#            environ = dict(SCRIPT_NAME='a',PATH_INFO='b')
            raise RESTException("404 NOT_FOUND", 
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

def wsgi_server(request_count=1):
    """ run this function from the command line in a terminal window """
    # invoke wsgi_app for each GET request
    httpd = make_server(host='localhost', port=8080, app=wsgi_app)
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
    
"""

# =============================================================================
# 
# =============================================================================
if __name__ == "__main__":
    wsgi_server(None) # serve forever

