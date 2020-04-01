# content of doc/autogen.py

from keras_autodoc import DocumentationGenerator


pages = {
    "layers/core.md": ["keras.layers.Dense", "keras.layers.Flatten"],
    "callbacks.md": ["keras.callbacks.TensorBoard"],
}

doc_generator = DocumentationGenerator(pages)
doc_generator.generate("./sources")
