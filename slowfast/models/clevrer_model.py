"""
Implemetation of the main Model for the Clevrer Dataset
It aggregates the Transformer and MONet models
"""

class ClevrerMain()

    
    def assemble_input(self):
        """
        Assembles the input sequence for the Transformer.
        Receives: slots, word embeddings
        Sequence: <CLS, slots, words>
        The slots and words are concatenated with a one hot that indicates
        if they are slots or words. => Sequence vectors are d + 2 dimensional
        """
        pass
