/**
 * @file module.c
 * @brief Python extension module initialization for GPT-OSS
 *
 * This file implements the entry point for the _metal Python extension module.
 * It initializes and registers the Model, Tokenizer, and Context Python types.
 *
 * The module is imported in Python as:
 * @code{.py}
 * from gptoss import _metal
 * # or more typically via the gptoss package wrapper
 * import gptoss
 * @endcode
 *
 * Module initialization performs the following:
 * 1. Prepares Python type objects (PyType_Ready)
 * 2. Creates the module object
 * 3. Adds type objects to the module namespace
 * 4. Returns the initialized module to Python
 *
 * Error handling: If any initialization step fails, all allocated resources
 * are properly cleaned up before returning NULL to Python.
 */
#include <Python.h>

#include "module.h"

/**
 * @brief Module-level functions table
 *
 * Currently the module exports only type objects (Model, Tokenizer, Context)
 * and no standalone functions. This table is required by Python's module
 * definition but contains only the NULL sentinel.
 */
static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/**
 * @brief Module definition structure for _metal extension
 *
 * Defines metadata for the Python module including:
 * - Module name: "_metal" (typically imported via gptoss package)
 * - Module docstring: Brief description of functionality
 * - Module state size: -1 indicates no per-module state
 * - Module methods: Currently empty (types only)
 */
static PyModuleDef metal_module = {
    PyModuleDef_HEAD_INIT,
    "_metal",                      /* Module name */
    "Local GPT-OSS inference",     /* Module docstring */
    -1,                            /* Module state size */
    module_methods                 /* Module methods */
};

/**
 * @brief Module initialization function called by Python import system
 *
 * This is the entry point when Python imports the _metal extension module.
 * It performs the following initialization sequence:
 *
 * 1. Initialize Model type: PyType_Ready(&PyGPTOSSModel_Type)
 * 2. Initialize Tokenizer type: PyType_Ready(&PyGPTOSSTokenizer_Type)
 * 3. Initialize Context type: PyType_Ready(&PyGPTOSSContext_Type)
 * 4. Create the module object
 * 5. Add all types to the module's namespace
 *
 * @return Initialized module object on success, NULL on failure
 *
 * Error handling: Uses goto error pattern to ensure proper cleanup of all
 * reference counts if initialization fails at any step. All INCREF'd objects
 * are XDECREF'd in the error handler.
 *
 * Thread safety: Called once during module import; subsequent imports return
 * the cached module object.
 */
PyMODINIT_FUNC PyInit__metal(void) {
    PyObject* module = NULL;
    PyObject* model_type = NULL;
    PyObject* tokenizer_type = NULL;
    PyObject* context_type = NULL;

    /* Initialize Model type - fills in type slots and makes it ready for instantiation */
    if (PyType_Ready(&PyGPTOSSModel_Type) < 0) {
        goto error;
    }
    model_type = (PyObject*) &PyGPTOSSModel_Type;
    Py_INCREF(model_type);  /* Module will steal a reference, so increment first */

    /* Initialize Tokenizer type */
    if (PyType_Ready(&PyGPTOSSTokenizer_Type) < 0) {
        goto error;
    }
    tokenizer_type = (PyObject*) &PyGPTOSSTokenizer_Type;
    Py_INCREF(tokenizer_type);  /* Module will steal a reference */

    /* Initialize Context type */
    if (PyType_Ready(&PyGPTOSSContext_Type) < 0) {
        goto error;
    }
    context_type = (PyObject*) &PyGPTOSSContext_Type;
    Py_INCREF(context_type);  /* Module will steal a reference */

    /* Create the module object */
    module = PyModule_Create(&metal_module);
    if (module == NULL) {
        goto error;
    }

    /* Add Model type to module namespace (steals reference) */
    if (PyModule_AddObject(module, "Model", model_type) < 0) {
        goto error;
    }

    /* Add Tokenizer type to module namespace (steals reference) */
    if (PyModule_AddObject(module, "Tokenizer", tokenizer_type) < 0) {
        goto error;
    }

    /* Add Context type to module namespace (steals reference) */
    if (PyModule_AddObject(module, "Context", context_type) < 0) {
        goto error;
    }

    return module;

error:
    /* Clean up on error - XDECREF safely handles NULL pointers */
    Py_XDECREF(context_type);
    Py_XDECREF(tokenizer_type);
    Py_XDECREF(model_type);
    Py_XDECREF(module);
    return NULL;
}
