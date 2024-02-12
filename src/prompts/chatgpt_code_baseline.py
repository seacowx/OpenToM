import AttitudeReasoner

class DetectAttitude():
    '''
    Read and comprehend the following short story. Then, answer the question that follows.
    '''

    def __init__(self) -> None:
        self.reasoner = AttitudeReasoner()

    def provide_narrative(self):
        '''
        function that assigns a narrative to the variable
        '''
        self.narrative = """
        {narrative}
        """

    def provide_question(self):
        '''
        function that assigns a question to the variable
        '''
        self.question = "{question}"

    def options(self):
        '''
        function that assigns options to the variable
        '''
        self.options = [
            'positive',
            'appreciation',
            'neutral',
            'negative'
        ]

    def deduce_answer(self):
        '''
        provide reasoner with narrative, question, and options, it will provide with the best answer
        '''
        answer = self.reasoner.deduce_answer(self.narrative, self.question, self.options)

        assert str(answer) == 
