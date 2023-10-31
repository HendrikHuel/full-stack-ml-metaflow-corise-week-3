from metaflow import FlowSpec, step, catch


class DivideByZeroFlow(FlowSpec):
    @step
    def start(self):
        self.divisors = [0, 1, 2]
        self.next(self.divide, foreach="divisors")

    @catch(var='compute_failed')
    @step
    def divide(self):
        self.res = 10 / self.input  # A
        self.next(self.join)

    @step
    def join(self, inputs):
        self.results = [inp.res if not inp.compute_failed else None for inp in inputs]
        print("results", self.results)
        self.next(self.end)

    @step
    def end(self):
        print("done!")


if __name__ == "__main__":
    DivideByZeroFlow()
