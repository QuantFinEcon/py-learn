""" 
This PEP proposes re-designing the buffer interface 
(PyBufferProcs function pointers) to improve the way 
Python allows memory sharing in Python 3.0

In particular, it is proposed that the character buffer
portion of the API be eliminated and the 
multiple-segment portion be re-designed 
in conjunction with allowing for strided memory to be 
shared. In addition, the new buffer interface will 
allow the sharing of any multi-dimensional 
nature of the memory and what data-format the memory contains.

This interface will allow any extension module to either 
create objects that share memory or create algorithms 
that use and manipulate raw memory from arbitrary objects that 
export the interface.

http://jakevdp.github.io/blog/2014/05/05/introduction-to-the-python-buffer-protocol/
"""




