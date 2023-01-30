class Statistic:

    def __init__(self, name, value, time):
        self.name = name
        self.value = value
        self.time = time

    def __repr__(self):
        return "(" + str(self.name) + ", " + str(self.time) + ")"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

