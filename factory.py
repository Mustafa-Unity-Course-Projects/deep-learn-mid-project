import random

from pydantic import BaseModel, Field


class FactoryState(BaseModel):
    employees_exist: bool
    brightness: float = Field(le=1.0, ge=0.0)
    temp: float = Field(ge=0, le=40)

    @classmethod
    def create_randomly(cls) -> "FactoryState":
        return cls(
            employees_exist=bool(random.getrandbits(1)),
            brightness=float(random.randrange(0, 100)) / 100.0,
            temp=float(random.randrange(0, 40)),
        )

    def to_list_light(self):
        return [self.employees_exist, self.brightness]

    def to_list_ac(self):
        return [self.employees_exist, self.temp]


class Factory:
    state: FactoryState

    def __init__(self):
        self.next_state()

    def next_state(self):
        self.state = FactoryState.create_randomly()
        return self.state

    def action_light(self, action) -> float:
        lights_on = bool(action[0][0])
        state = self.state
        light_required = (state.employees_exist and state.brightness < 0.5)

        if lights_on and light_required:
            return 5.0
        if lights_on and not light_required:
            return -3.0
        if not lights_on and light_required:
            return -5.0
        if not lights_on and not light_required:
            return 2.0

    def action_ac(self, action) -> float:
        ac_on = bool(action[0][0])
        state = self.state
        ac_required = (state.employees_exist and state.temp > 25)

        if ac_on and ac_required:
            return 5.0
        if ac_on and not ac_required:
            return -3.0
        if not ac_on and ac_required:
            return -5.0
        if not ac_on and not ac_required:
            return 2.0
