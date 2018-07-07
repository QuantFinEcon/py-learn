from cgi import parse_qs
from wsgiref.simple_server import make_server

def simple_app(environ, start_response):
    status = '200 OK'
    headers = [('Content-Type', 'text/plain')]
    start_response(status, headers)
    if environ['REQUEST_METHOD'] == 'POST':
        request_body_size = int(environ.get('CONTENT_LENGTH', 0))
        request_body = environ['wsgi.input'].read(request_body_size)
        d = parse_qs(request_body)  # turns the qs to a dict
        return 'From POST: %s' % ''.join('%s: %s' % (k, v) for k, v in d.iteritems())
    else:  # GET
        d = parse_qs(environ['QUERY_STRING'])  # turns the qs to a dict
        return 'From GET: %s' % ''.join('%s: %s' % (k, v) for k, v in d.iteritems())

httpd = make_server('', 8080, simple_app)
print("Serving on port 8080...")
httpd.serve_forever()

#
#$ python server.py 
#Serving on port 8080...
#1.0.0.127.in-addr.arpa - - [25/Oct/2011 10:36:10] "POST / HTTP/1.1" 200 24
#1.0.0.127.in-addr.arpa - - [25/Oct/2011 10:36:11] "GET /?foo=bar HTTP/1.1" 200 22
#
#$ curl http://localhost:8080/?foo=bar
#From GET: foo: ['bar']
#$ curl -d baz=quux http://localhost:8080/
#From POST: baz: ['quux']