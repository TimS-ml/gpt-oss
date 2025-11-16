/**
 * @file module.h
 * @brief Internal header for GPT-OSS Python C extension module
 *
 * This header defines the Python object structures that wrap the C API handles,
 * providing Python bindings for the GPT-OSS library. Each Python object contains
 * the standard PyObject_HEAD and an opaque handle to the corresponding C object.
 *
 * Python Module Structure:
 * - gptoss._metal: Native extension module
 * - gptoss.Model: Python wrapper for gptoss_model_t
 * - gptoss.Tokenizer: Python wrapper for gptoss_tokenizer_t
 * - gptoss.Context: Python wrapper for gptoss_context_t
 *
 * Memory Management:
 * Python objects automatically retain/release their C handles using Python's
 * reference counting mechanism. When the Python object is deallocated, the
 * corresponding C object is released.
 */
#include <Python.h>

#include <gpt-oss.h>

/**
 * @struct PyGPTOSSModel
 * @brief Python object wrapping a gptoss_model_t handle
 *
 * This structure represents a Model object in Python. It contains the standard
 * PyObject header (for Python's object system) and a handle to the underlying
 * C model object.
 *
 * Python usage:
 * @code{.py}
 * model = gptoss.Model("path/to/model.bin")
 * max_ctx = model.max_context_length
 * tokenizer = model.tokenizer
 * @endcode
 */
typedef struct {
    PyObject_HEAD
    gptoss_model_t handle;  /* Handle to the C model object */
} PyGPTOSSModel;

/**
 * @struct PyGPTOSSTokenizer
 * @brief Python object wrapping a gptoss_tokenizer_t handle
 *
 * This structure represents a Tokenizer object in Python. It wraps the C
 * tokenizer handle and provides methods for token encoding/decoding.
 *
 * Python usage:
 * @code{.py}
 * tokenizer = model.tokenizer
 * token_id = tokenizer.encode_special_token("<|start|>")
 * text_bytes = tokenizer.decode(token_id)
 * @endcode
 */
typedef struct {
    PyObject_HEAD
    gptoss_tokenizer_t handle;  /* Handle to the C tokenizer object */
} PyGPTOSSTokenizer;

/**
 * @struct PyGPTOSSContext
 * @brief Python object wrapping a gptoss_context_t handle
 *
 * This structure represents an inference Context in Python. It maintains the
 * token sequence and KV cache state for generation.
 *
 * Python usage:
 * @code{.py}
 * ctx = gptoss.Context(model, context_length=2048)
 * ctx.append("Hello, world!")
 * ctx.process()
 * tokens = ctx.sample(max_output_tokens=10, temperature=0.8)
 * @endcode
 */
typedef struct {
    PyObject_HEAD
    gptoss_context_t handle;  /* Handle to the C context object */
} PyGPTOSSContext;

/* Python type objects for the extension module */
extern PyTypeObject PyGPTOSSModel_Type;
extern PyTypeObject PyGPTOSSTokenizer_Type;
extern PyTypeObject PyGPTOSSContext_Type;
