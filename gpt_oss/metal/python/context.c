/**
 * @file context.c
 * @brief Python bindings for GPT-OSS Context object
 *
 * This file implements the Python wrapper for the gptoss_context_t C API.
 * The PyGPTOSSContext type provides a Pythonic interface to:
 * - Create inference contexts with configurable sizes
 * - Append text (strings/bytes) or tokens to the context
 * - Process tokens through the model (prefill/decode)
 * - Sample next tokens with temperature control
 * - Query context state (tokens, capacity)
 * - Reset context for reuse
 *
 * Python usage example:
 * @code{.py}
 * ctx = gptoss.Context(model, context_length=2048, max_batch_tokens=512)
 * ctx.append("Hello, ")
 * ctx.process()
 * tokens = ctx.sample(max_output_tokens=10, temperature=0.8, seed=42)
 * # Decode and append tokens back to context for multi-turn generation
 * @endcode
 *
 * Memory management: The Python object manages the C context handle with
 * reference counting. The KV cache memory is allocated on the GPU.
 */
#include <Python.h>

#include <gpt-oss.h>

#include "module.h"

/**
 * @brief Initialize a new Context object
 *
 * Called when creating a Context instance from Python. Allocates a context
 * for the given model with specified capacity constraints.
 *
 * @param self The Python Context object being initialized
 * @param args Positional arguments: (model,)
 * @param kwargs Keyword arguments: context_length, max_batch_tokens
 * @return 0 on success, -1 on failure
 *
 * Python signature:
 * Context(model: Model, *, context_length: int = 0, max_batch_tokens: int = 0)
 *
 * Arguments:
 * - model: The Model object to use for inference
 * - context_length: Maximum tokens in context (0 = use model's maximum)
 * - max_batch_tokens: Maximum tokens per batch (0 = use default)
 *
 * Error handling: Sets Python ValueError for invalid arguments, generic
 * exception for API failures.
 */
static int PyGPTOSSContext_init(PyGPTOSSContext* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"model", "context_length", "max_batch_tokens", NULL};
    PyObject* model = NULL;
    Py_ssize_t context_length = 0; // Default to 0 if None
    Py_ssize_t max_batch_tokens = 0; // Default to 0 if None

    /* Parse arguments: required model, optional keyword-only length/batch size */
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|$ii", kwlist,
                                     &PyGPTOSSModel_Type, &model,
                                     &context_length, &max_batch_tokens))
    {
        return -1;
    }

    /* Validate argument ranges */
    if (context_length < 0) {
        PyErr_SetString(PyExc_ValueError, "context_length must be a positive integer");
        return -1;
    }
    if (max_batch_tokens < 0) {
        PyErr_SetString(PyExc_ValueError, "max_batch_tokens must be a positive integer");
        return -1;
    }

    /* Create C context using model's handle */
    enum gptoss_status status = gptoss_context_create(
        ((const PyGPTOSSModel*) model)->handle,
        (size_t) context_length,
        (size_t) max_batch_tokens,
        &self->handle);
    if (status != gptoss_status_success) {
        // TODO: set exception with proper error message
        goto error;
    }

    return 0;

error:
    gptoss_context_release(self->handle);
    self->handle = NULL;
    return -1;
}

/**
 * @brief Deallocate a Context object
 *
 * Releases the C context handle (freeing KV cache GPU memory if this is
 * the last reference) and frees the Python object.
 */
static void PyGPTOSSContext_dealloc(PyGPTOSSContext* self) {
    (void) gptoss_context_release(self->handle);
    self->handle = NULL;
    PyObject_Del((PyObject*) self);
}

/**
 * @brief Create a shallow copy of the Context
 *
 * Creates a new Python object sharing the same C context handle.
 * Both copies will reference the same KV cache state.
 */
static PyObject* PyGPTOSSContext_copy(PyGPTOSSContext *self) {
    PyGPTOSSContext* copy = (PyGPTOSSContext*) PyObject_New(PyGPTOSSContext, Py_TYPE(self));
    if (copy == NULL) {
        return NULL;
    }

    (void) gptoss_context_retain(self->handle);
    copy->handle = self->handle;
    return (PyObject*) copy;
}

/**
 * @brief Append data to the context
 *
 * Polymorphic method that accepts:
 * - bytes: Raw UTF-8 bytes to tokenize and append
 * - str: Unicode string to encode as UTF-8, tokenize, and append
 * - int: Single token ID to append directly
 *
 * @param self The Context object
 * @param arg Data to append (bytes, str, or int)
 * @return None on success, NULL on error
 *
 * Python usage:
 * @code{.py}
 * ctx.append("Hello")      # String
 * ctx.append(b"world")     # Bytes
 * ctx.append(12345)        # Token ID
 * @endcode
 */
static PyObject* PyGPTOSSContext_append(PyGPTOSSContext* self, PyObject* arg) {
    if (PyBytes_Check(arg)) {
        /* Handle bytes: tokenize and append */
        char* string_ptr = NULL;
        Py_ssize_t string_size = 0;
        if (PyBytes_AsStringAndSize(arg, &string_ptr, &string_size) < 0) {
            return NULL;
        }

        const enum gptoss_status status = gptoss_context_append_chars(
            self->handle, string_ptr, string_size, /*num_tokens_out=*/NULL);
        if (status != gptoss_status_success) {
            // TODO: set exception
            return NULL;
        }

        Py_RETURN_NONE;
    } else if (PyUnicode_Check(arg)) {
        /* Handle string: encode to UTF-8, tokenize, and append */
        Py_ssize_t string_size = 0;
        const char* string_ptr = PyUnicode_AsUTF8AndSize(arg, &string_size);
        if (string_ptr == NULL) {
            return NULL;
        }

        const enum gptoss_status status = gptoss_context_append_chars(
            self->handle, string_ptr, string_size, /*num_tokens_out=*/NULL);
        if (status != gptoss_status_success) {
            // TODO: set exception
            return NULL;
        }

        Py_RETURN_NONE;
    } else if (PyLong_Check(arg)) {
        /* Handle int: append token ID directly */
        const unsigned long token_as_ulong = PyLong_AsUnsignedLong(arg);
        if (token_as_ulong == (unsigned long) -1 && PyErr_Occurred()) {
            return NULL;
        }

        const uint32_t token = (uint32_t) token_as_ulong;
        const enum gptoss_status status = gptoss_context_append_tokens(
            self->handle, /*num_tokens=*/1, &token);
        if (status != gptoss_status_success) {
            // TODO: set exception
            return NULL;
        }

        Py_RETURN_NONE;
    } else {
        PyErr_SetString(PyExc_TypeError, "expected a bytes or integer argument");
        return NULL;
    }
}

/**
 * @brief Process tokens through the model
 *
 * Runs inference on all unprocessed tokens in the context, updating the
 * KV cache and generating logits for the next token prediction.
 *
 * @param self The Context object
 * @return None on success, NULL on error
 *
 * Python usage: ctx.process()
 *
 * This performs prefill (for multiple tokens) or decode (for single token).
 * Must be called after append() and before sample().
 */
static PyObject* PyGPTOSSContext_process(PyGPTOSSContext* self) {
    const enum gptoss_status status = gptoss_context_process(self->handle);
    if (status != gptoss_status_success) {
        // TODO: set exception
        return NULL;
    }

    Py_RETURN_NONE;
}

/**
 * @brief Sample next tokens from the model
 *
 * Samples one or more tokens from the model's output distribution using
 * the specified temperature and random seed.
 *
 * @param self The Context object
 * @param args Positional arguments: (max_output_tokens,)
 * @param kwargs Keyword arguments: temperature, seed
 * @return List of sampled token IDs, or NULL on error
 *
 * Python signature:
 * sample(max_output_tokens: int, *, temperature: float = 1.0, seed: int = 0)
 *
 * Arguments:
 * - max_output_tokens: Maximum number of tokens to generate
 * - temperature: Sampling temperature (default 1.0, 0.0 for greedy)
 * - seed: Random seed for reproducibility (default 0)
 *
 * Returns: List of token IDs (may be shorter than max_output_tokens if
 * generation stops early)
 *
 * Must be called after process().
 */
static PyObject* PyGPTOSSContext_sample(PyGPTOSSContext* self, PyObject* args, PyObject* kwargs) {
    static char *kwlist[] = {"max_output_tokens", "temperature", "seed", NULL};
    PyObject* token_list_obj = NULL;
    uint32_t* token_ptr = NULL;

    unsigned int max_output_tokens = 0;
    unsigned long long seed = 0;
    float temperature = 1.0f;

    /* Parse arguments: required max tokens, optional temperature and seed */
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "I|$fK", kwlist,
            &max_output_tokens, &temperature, &seed))
    {
        return NULL;
    }

    /* Allocate temporary buffer for sampled token IDs */
    token_ptr = (uint32_t*) PyMem_Malloc(max_output_tokens * sizeof(uint32_t));
    if (token_ptr == NULL) {
        goto error;
    }

    /* Call C API to sample tokens */
    size_t num_tokens = 0;
    const enum gptoss_status status = gptoss_context_sample(
        self->handle, temperature, (uint64_t) seed,
        (size_t) max_output_tokens, token_ptr, &num_tokens);
    if (status != gptoss_status_success) {
        // TODO: set exception
        goto error;
    }

    /* Create Python list to hold results */
    token_list_obj = PyList_New((Py_ssize_t) num_tokens);
    if (token_list_obj == NULL) {
        goto error;
    }

    /* Convert C token IDs to Python ints and add to list */
    for (size_t t = 0; t < num_tokens; t++) {
        PyObject* token_obj = PyLong_FromUnsignedLong((unsigned long) token_ptr[t]);
        if (token_obj == NULL) {
            goto error;
        }

        PyList_SET_ITEM(token_list_obj, (Py_ssize_t) t, token_obj);
    }

    PyMem_Free(token_ptr);
    return token_list_obj;

error:
    PyMem_Free(token_ptr);
    Py_XDECREF(token_list_obj);
    return NULL;
}

/**
 * @brief Reset the context to empty state
 *
 * Clears all tokens and KV cache state, allowing the context to be
 * reused for a new sequence.
 *
 * @param self The Context object
 * @return None on success, NULL on error
 *
 * Python usage: ctx.reset()
 *
 * This is more efficient than creating a new context as it reuses
 * allocated GPU memory.
 */
static PyObject* PyGPTOSSContext_reset(PyGPTOSSContext* self) {
    const enum gptoss_status status = gptoss_context_reset(self->handle);
    if (status != gptoss_status_success) {
        // TODO: set exception
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyMethodDef PyGPTOSSContext_methods[] = {
    {"__copy__", (PyCFunction) PyGPTOSSContext_copy, METH_NOARGS, "Create a copy of the Context"},
    {"append", (PyCFunction) PyGPTOSSContext_append, METH_O, "Append bytes to the Context"},
    {"process", (PyCFunction) PyGPTOSSContext_process, METH_NOARGS, "Process tokens in the Context"},
    {"sample", (PyCFunction) PyGPTOSSContext_sample, METH_VARARGS | METH_KEYWORDS, "Sample token predictions from the Context"},
    {"reset", (PyCFunction) PyGPTOSSContext_reset, METH_NOARGS, "Discard the content of the Context"},
    {NULL},
};

static PyObject* PyGPTOSSContext_get_num_tokens(PyGPTOSSContext* self, void* closure) {
    size_t num_tokens = 0;
    const enum gptoss_status status = gptoss_context_get_num_tokens(self->handle, &num_tokens);
    if (status != gptoss_status_success) {
        // TODO: set exception
        return NULL;
    }

    return PyLong_FromSize_t(num_tokens);
}

static PyObject* PyGPTOSSContext_get_max_tokens(PyGPTOSSContext* self, void* closure) {
    size_t max_tokens = 0;
    const enum gptoss_status status = gptoss_context_get_max_tokens(self->handle, &max_tokens);
    if (status != gptoss_status_success) {
        // TODO: set exception
        return NULL;
    }

    return PyLong_FromSize_t(max_tokens);
}

static PyObject* PyGPTOSSContext_get_tokens(PyGPTOSSContext* self, void* closure) {
    PyObject* token_list_obj = NULL;
    uint32_t* token_ptr = NULL;

    size_t num_tokens = 0;
    gptoss_context_get_tokens(self->handle, /*tokens_out=*/NULL, /*max_tokens=*/0, &num_tokens);

    if (num_tokens != 0) {
        token_ptr = (uint32_t*) PyMem_Malloc(num_tokens * sizeof(uint32_t));
        if (token_ptr == NULL) {
            // TODO: set exception
            goto error;
        }

        enum gptoss_status status = gptoss_context_get_tokens(self->handle, token_ptr, /*max_tokens=*/num_tokens, &num_tokens);
        if (status != gptoss_status_success) {
            // TODO: set exception
            goto error;
        }
    }

    token_list_obj = PyList_New((Py_ssize_t) num_tokens);
    if (token_list_obj == NULL) {
        goto error;
    }

    for (size_t t = 0; t < num_tokens; t++) {
        PyObject* token_obj = PyLong_FromUnsignedLong((unsigned long) token_ptr[t]);
        if (token_obj == NULL) {
            goto error;
        }

        PyList_SET_ITEM(token_list_obj, (Py_ssize_t) t, token_obj);
    }

    PyMem_Free(token_ptr);
    return token_list_obj;

error:
    PyMem_Free(token_ptr);
    Py_XDECREF(token_list_obj);
    return NULL;
}

static PyGetSetDef PyGPTOSSContext_getseters[] = {
    (PyGetSetDef) {
        .name = "num_tokens",
        .get = (getter) PyGPTOSSContext_get_num_tokens,
        .doc = "Current number of tokens in the context",
    },
    (PyGetSetDef) {
        .name = "max_tokens",
        .get = (getter) PyGPTOSSContext_get_max_tokens,
        .doc = "Maximum number of tokens in the context",
    },
    (PyGetSetDef) {
        .name = "tokens",
        .get = (getter) PyGPTOSSContext_get_tokens,
        .doc = "List of token IDs in the context",
    },
    {NULL}  /* Sentinel */
};

PyTypeObject PyGPTOSSContext_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "gptoss.Context",
    .tp_basicsize = sizeof(PyGPTOSSContext),
    .tp_flags = 0
        | Py_TPFLAGS_DEFAULT
        | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Context object",
    .tp_methods = PyGPTOSSContext_methods,
    .tp_getset = PyGPTOSSContext_getseters,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) PyGPTOSSContext_init,
    .tp_dealloc = (destructor) PyGPTOSSContext_dealloc,
};
