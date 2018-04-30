/* https://docs.python.org/3/c-api/ */
/* WRAPPER */

/* python wrapper to chi2.c using special python functions 
    to access python types
   
    C doesn't have any concept of namespaces, the 
    convention is to name your C functions with 
    the form {module_name}_{function_name}.. so 
    conventionally naming starts with _ for calling 
    Example: init_chi2

    Note: wrappers like pyximport, cython ensures you don't directly 
    interact with tht python/C API

    all that I need to do is wrap a single C function that 
    accepts a few doubles and returns another double
*/

// import python headers
#include <Python.h>
#include <numpy/arrayobject.h>
#include "chi2.h"

// doctring for module and the func we are wrapping
static char module_docstring[] =
    "This module provides an interface for calculating chi-squared using C.";
static char chi2_docstring[] = 
    "Calculate the chi-squared of some data given a model.";

// declare function, PyObject refers to all Python types.

/* https://docs.python.org/3/c-api/ */
/* Python/C API INTERFACE */
static PyObject *chi2_chi2(PyObject *self, PyObject *args)
{
    double m, b;
    PyObject *x_obj, *y_obj, *yerr_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "ddOOO", &m, &b, &x_obj, &y_obj,
                                        &yerr_obj))
        return NULL;
    /* 
    PyArg_ParseTuple

    This function takes the tuple, a format and the list of 
    pointers to the objects that you want to take the input values. 
    This format should be familiar if you've ever used something like 
    the sscanf function in C but the format characters are a little 
    different. 
    
    In our example, d indicates that the argument should 
    be cast as a C double and O is just a catchall for PyObjects. 
    There isn't a specific format character for numpy arrays so we have 
    to parse them as raw PyObjects and then interpret them afterwards. 
    
    If PyArg_ParseTuple fails, it will return NULL which is 
    the C-API technique for propagating exceptions.
    */


    /* Interpret the raw input objects as numpy arrays. */
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *yerr_array = PyArray_FROM_OTF(yerr_obj, NPY_DOUBLE,
                                            NPY_IN_ARRAY);
    /*
    PyArray_FROM_OTF: converting an arbitrary Python object into a 
    well-behaved numpy array that can be used in a standard C function.
 
    NPY_DOUBLE and NPY_IN_ARRAY ensure that the returned array object 
    will be represented as contiguous arrays of C doubles
    */


    /* If that didn't work, throw an exception. */
    if (x_array == NULL || y_array == NULL || yerr_array == NULL) {
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        Py_XDECREF(yerr_array);
        return NULL;
    }
    /*
    when you return a PyObject from a function you might want to run 
    Py_INCREF on it to increment the reference count and when you 
    create a new object within you function that you don't want to 
    return, you should run Py_DECREF on it before the function 
    returns (even if the execution failed) so that you don't
    end up with a memory leak.

    Py_XDECREF checks to make sure that the object 
    isn't a NULL pointer before trying to decrease the 
    reference count whereas Py_DECREF will explode 
    if you try to call it on NULL.

    it is part of each function's "interface specification" 
    whether or not it increases the reference count of an object 
    before it returns or not. 
    */

    /* How many data points are there? */
    int N = (int)PyArray_DIM(x_array, 0);

    /* Get pointers to the data as C-types. */
    double *x    = (double*)PyArray_DATA(x_array);
    double *y    = (double*)PyArray_DATA(y_array);
    double *yerr = (double*)PyArray_DATA(yerr_array);

    /* Finally, after inputs, refcounts, and linking the pointers */

    /* Call the external chi2.c C function to compute the chi-squared. */
    double value = chi2(m, b, x, y, yerr, N);

    /* Clean up. Free memory by reducing refcount to 0. */
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(yerr_array);

    /* throw an exception if something went wrong in 
    the execution of the C code */
    if (value < 0.0) {
        PyErr_SetString(PyExc_RuntimeError,
                    "Chi-squared returned an impossible value.");
        return NULL;
    }
    /*
    Python interpreter has a global variable that contains a 
    pointer to the most recent exception that has been thrown. 

    Then if a function returns NULL it starts an upwards cascade 
    where each function either catches the exception using a 
    try-except statement or also returns NULL. 
    
    When the interpreter receives a NULL return value, it stops 
    execution of the current code and shows a representation 
    of the value of the global Exception variable and the traceback.
    */


    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", value);
    return ret;

    /*
    If Py_ParseTuple has a syntax similar to sscanf then 
    Py_BuildValue is the analog of sprintf with 
    the same format characters as Py_ParseTuple. 
    */
}
/* Any communication between the Python interpreter and your C code
 will be done by passing PyObjects so any function that you
  want to be able to call from Python must return one. 
  Under the hood, PyObject is just a struct with a reference 
  count and a pointer to the data contained within the object.
*/

// specify what the members of this module will be. 
static PyMethodDef module_methods[] = {
    {"chi2", chi2_chi2, METH_VARARGS, chi2_docstring},
    {NULL, NULL, 0, NULL}
};

/* second line contains all the info that the interpreter 
needs to link a Python call to the correct C function 
and call it in the right way. 

what python needs to call 

{name of function called from python, 
C function python is links to, 
METH_VARARGS/METH_KEYWORDS means that the function 
only accepts positional/keyword arguments,
docstring for the function}   */

// final step in initializing your new C module
// is to write an init{name} function where _chi2 is name of module
// Py_InitModule3(name, methods, docstring)
PyMODINIT_FUNC init_chi2(void)
{
    PyObject *m = Py_InitModule3("_chi2", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. 
    import_array() (a function defined in the numpy/arrayobject.h header).
    */
    import_array();
}



