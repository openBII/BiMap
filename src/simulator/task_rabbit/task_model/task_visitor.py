class TaskVisitor():
    """
    TaskVisitor类为访问任务块的访问者
    """

    def __init__(self):
        pass

    def visit(self, task):
        pass

    def visit_S(self, task):
        self.visit(task)

    def visit_C(self, task):
        self.visit(task)

    def visit_SIC(self, task):
        self.visit_S(task)

    def visit_SIC2D(self, task):
        self.visit_S(task)

    def visit_SIFC(self, task):
        self.visit_S(task)

    def visit_SI(self, task):
        self.visit_S(task)

    def visit_SW(self, task):
        self.visit_S(task)

    def visit_SB(self, task):
        self.visit_S(task)

    def visit_SWFC(self, task):
        self.visit_S(task)

    def visit_SW2D(self, task):
        self.visit_S(task)

    def visit_CADD(self, task):
        self.visit_C(task)

    def visit_CVVH(self, task):
        self.visit_C(task)

    def visit_CVM(self, task):
        self.visit_C(task)

    def visit_CC(self, task):
        self.visit_C(task)

    def visit_CAX(self, task):
        self.visit_C(task)

    def visit_CC2D(self, task):
        self.visit_C(task)

    def visit_CVS(self, task):
        self.visit_C(task)

    def visit_CCMPB(self, task):
        self.visit_C(task)

    def visit_CCMPS(self, task):
        self.visit_C(task)

    def visit_CLUT(self, task):
        self.visit_C(task)

    def visit_CLIF(self, task):
        self.visit_C(task)

    def visit_CAVG(self, task):
        self.visit_C(task)

    def visit_OUTPUT(self, task):
        self.visit(task)

    def visit_INPUT(self, task):
        self.visit(task)
