
import sys
import wsgiref.util
from wsgiref.simple_server import make_server
import http.client
import json
import concurrent.futures
import time
from collections.abc import Callable

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
        """ handle wsgi processing: evaluation and response """
        response = self.run() # 3. Evaluate.
        status = '200 OK' # 4. Respond.
        headers = [('Content-type', 'application/json; charset=utf-8')]
        start_response(status, headers)
        return [ json.dumps(response).encode('UTF-8') ]
    
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
class RESTException(Exception):
    pass

def wsgi_app(environ, start_response):
    """ application to host on server """
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
            if request.lower()=='transact': # 3 + 4 wrapped downstream
                response= self.transact(environ,start_response)
            elif request.lower()=='transact2':
                response= self.transact2(environ,start_response)
            else:
                # not a valid request URI
                pass
            
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

# =============================================================================
# GET request to wsgi-server
# =============================================================================
def json_get(path="/"):
    """ Only GET reuqest client with no POST """
    # transaction group: create HTTPConnection for each request
    rest= http.client.HTTPConnection(host='localhost', port=8080) # 1. connect to server
    rest.request("GET", path) # 2. send request

    response= rest.getresponse() # 3. get response
    print( response.status, response.reason )
    print( response.getheaders() )

    # decode serialised byte transfer via HTTP
    raw= response.read().decode("utf-8") # 4. read response
    if response.status == 200:
        document= json.loads(raw)
        print( document )
    else:
        print( raw )

# =============================================================================
# simulate running of server
# =============================================================================
"""
start wsgi-server with if __name__=="__main__": in another python thread terminal
$ python wsgi-server.py
"""

with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.submit( wsgi_server, request_count=4 )
    time.sleep(2) # Wait for the server to start
    json_get("/transact/")
    json_get("/transact/")
    json_get() # raise RESTException in app1

#abstract base class for concrete asynchronous executors
executor=concurrent.futures.ProcessPoolExecutor()
executor._max_workers
executor._processes
executor._queue_count
executor._pending_work_items
executor.shutdown()







