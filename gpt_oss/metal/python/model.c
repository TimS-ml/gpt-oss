/**
 * @file model.c
 * @brief Python bindings for GPT-OSS Model object
 *
 * This file implements the Python wrapper for the gptoss_model_t C API.
 * The PyGPTOSSModel type provides a Pythonic interface to:
 * - Load models from files
 * - Query model properties (max context length)
 * - Access the associated tokenizer
 * - Copy model objects (reference counting)
 *
 * Python usage example:
 * @code{.py}
 * model = gptoss.Model("path/to/model.bin")
 * print(f"Max context: {model.max_context_length}")
 * tokenizer = model.tokenizer
 * model_copy = copy.copy(model)  # Shares underlying C object
 * @endcode
 *
 * Memory management: The Python object automatically retains the C handle
 * on creation and releases it on deallocation. Copying increments the
 * reference count rather than duplicating the model data.
 */
#include <Python.h>

#include <gpt-oss.h>

#include "module.h"

/**
 * @brief Initialize a new Model object
 *
 * Called when creating a Model instance from Python. Loads the model from
 * a file and stores the handle in the Python object.
 *
 * @param self The Python Model object being initialized
 * @param args Positional arguments: (filepath,)
 * @param kwargs Keyword arguments (currently unused)
 * @return 0 on success, -1 on failure
 *
 * Python signature: Model(filepath: str)
 *
 * Error handling: Sets Python exception and returns -1 if:
 * - Argument parsing fails
 * - Model file cannot be loaded
 * - Model format is invalid
 */
static int PyGPTOSSModel_init(PyGPTOSSModel* self, PyObject* args, PyObject* kwargs) {
    enum gptoss_status status;
    const char* filepath;

    /* Parse filepath argument from Python */
    if (!PyArg_ParseTuple(args, "s", &filepath)) {
        return -1;
    }

    /* Load model from file using C API */
    status = gptoss_model_create_from_file(filepath, &self->handle);
    if (status != gptoss_status_success) {
        // TODO: set exception with proper error message based on status
        return -1;
    }
    return 0;
}

/**
 * @brief Deallocate a Model object
 *
 * Called when Python's garbage collector destroys the Model object.
 * Releases the underlying C model handle (decrements reference count)
 * and frees the Python object memory.
 *
 * @param self The Model object being deallocated
 */
static void PyGPTOSSModel_dealloc(PyGPTOSSModel* self) {
    /* Release the C model handle (may free resources if last reference) */
    (void) gptoss_model_release(self->handle);
    self->handle = NULL;
    PyObject_Del((PyObject*) self);
}

/**
 * @brief Create a shallow copy of the Model object
 *
 * Implements the __copy__ method for Python's copy.copy() function.
 * Creates a new Python object but shares the underlying C model handle
 * by incrementing its reference count. This is efficient since model
 * weights are immutable and can be safely shared.
 *
 * @param self The Model object to copy
 * @return New Model object sharing the same C handle, or NULL on error
 *
 * Python usage: copy.copy(model) or model.__copy__()
 */
static PyObject* PyGPTOSSModel_copy(PyGPTOSSModel* self) {
    /* Allocate new Python object */
    PyGPTOSSModel* copy = (PyGPTOSSModel*) PyObject_New(PyGPTOSSModel, Py_TYPE(self));
    if (copy == NULL) {
        return NULL;
    }

    /* Retain C handle and share it with the new Python object */
    (void) gptoss_model_retain(self->handle);
    copy->handle = self->handle;
    return (PyObject*) copy;
}

/**
 * @brief Method table for Model objects
 *
 * Defines methods callable on Model instances from Python.
 * Currently only supports __copy__ for shallow copying.
 */
static PyMethodDef PyGPTOSSModel_methods[] = {
    {"__copy__", (PyCFunction) PyGPTOSSModel_copy, METH_NOARGS, "Create a copy of the Model"},
    {NULL},  /* Sentinel */
};

/**
 * @brief Get the maximum context length property
 *
 * Implements the max_context_length property getter. Queries the C API
 * for the maximum number of tokens this model can process.
 *
 * @param self The Model object
 * @param closure Unused (required by getter signature)
 * @return Python int with max context length, or NULL on error
 *
 * Python usage: model.max_context_length
 */
static PyObject *PyGPTOSSModel_get_max_context_length(PyGPTOSSModel* self, void* closure) {
    size_t max_context_length = 0;

    /* Query C API for max context length */
    const enum gptoss_status status = gptoss_model_get_max_context_length(self->handle, &max_context_length);
    if (status != gptoss_status_success) {
        // TODO: set exception with proper error message
        return NULL;
    }

    /* Convert to Python int */
    return PyLong_FromSize_t(max_context_length);
}

/**
 * @brief Get the tokenizer property
 *
 * Implements the tokenizer property getter. Creates a new Tokenizer Python
 * object wrapping the C tokenizer handle associated with this model.
 *
 * @param self The Model object
 * @param closure Unused (required by getter signature)
 * @return New Tokenizer object, or NULL on error
 *
 * Python usage: tokenizer = model.tokenizer
 *
 * Note: The Tokenizer's __init__ handles querying the C API and retaining
 * the tokenizer handle.
 */
static PyObject *PyGPTOSSModel_get_tokenizer(PyGPTOSSModel* self, void* closure) {
    /* Create argument tuple containing this model */
    PyObject* args = PyTuple_Pack(1, self);
    if (args == NULL) {
        return NULL;
    }

    /* Call Tokenizer constructor with model argument */
    PyObject* tokenizer = PyObject_CallObject((PyObject*) &PyGPTOSSTokenizer_Type, args);
    Py_DECREF(args);
    return tokenizer;
}

/**
 * @brief Property definitions for Model objects
 *
 * Defines properties accessible via attribute access in Python.
 * Properties are read-only (no setter defined).
 */
static PyGetSetDef PyGPTOSSModel_getseters[] = {
    (PyGetSetDef) {
        .name = "max_context_length",
        .get = (getter) PyGPTOSSModel_get_max_context_length,
        .doc = "Maximum context length supported by the model",
    },
    (PyGetSetDef) {
        .name = "tokenizer",
        .get = (getter) PyGPTOSSModel_get_tokenizer,
        .doc = "Tokenizer object associated with the model",
    },
    {NULL}  /* Sentinel */
};

/**
 * @brief Type definition for Model objects
 *
 * Defines the Python type gptoss.Model. This structure specifies:
 * - Type name and size
 * - Type flags (allows subclassing, uses default behavior)
 * - Documentation string
 * - Methods table
 * - Properties table
 * - Lifecycle functions (new, init, dealloc)
 *
 * This type is registered with Python during module initialization.
 */
PyTypeObject PyGPTOSSModel_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "gptoss.Model",
    .tp_basicsize = sizeof(PyGPTOSSModel),
    .tp_flags = 0
        | Py_TPFLAGS_DEFAULT      /* Use default type behavior */
        | Py_TPFLAGS_BASETYPE,    /* Allow subclassing */
    .tp_doc = "Model object",
    .tp_methods = PyGPTOSSModel_methods,
    .tp_getset = PyGPTOSSModel_getseters,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) PyGPTOSSModel_init,
    .tp_dealloc = (destructor) PyGPTOSSModel_dealloc,
};
