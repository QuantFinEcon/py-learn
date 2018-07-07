
import sys
import wsgiref.util
from wsgiref.simple_server import make_server
import http.client
import json
import concurrent.futures
import time

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

# =============================================================================
# GET request to wsgi-server
# =============================================================================
def json_get(path="/"):
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







