import random

import pygame
import pygame_gui
import torch

from factory import FactoryState
from model import DQN

COLOR_YELLOW = pygame.Color(253, 250, 114)
COLOR_RED = pygame.Color(255, 50, 50)

pygame.init()
pygame.font.init()
pygame.display.set_caption('Quick Start')
window_surface = pygame.display.set_mode((1200, 600))
background = pygame.Surface((1200, 600))
background.fill(pygame.Color('#000000'))
manager = pygame_gui.UIManager((1200, 600))
my_font = pygame.font.SysFont("Arial", 30)

worker_sprite = pygame.image.load("worker.jpg")

is_running = True
workday = True

minute = 0
minute_mod = 60 * 24

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
light_model = DQN(2, 2).to(device)
light_model.load_state_dict(torch.load("light_model_weights"))
ac_model = DQN(2, 2).to(device)
ac_model.load_state_dict(torch.load("ac_model_weights"))


def mins_to_time(minutes: int) -> str:
    hours, mins = minutes // 60, minutes % 60
    return f"{hours:02}:{mins:02}"

factoryState = FactoryState(brightness=0.0, temp=20.0, employees_exist=False)

def list_to_tensor(list_: list):
    return torch.tensor(list_, dtype=torch.float32, device=device).unsqueeze(0)


def mins(hour: int) -> int:
    return hour * 60

def get_ratio(val, min_, max_) -> float:
    if val > max_:
        raise Exception("val bigger than max. Invalid ratio call.")
    return float(val - min_) / float(max_ - min_)

def get_brightness(minute: int) -> float:
    if minute < mins(5) or minute > mins(18):
        return 0.0
    elif mins(5) <= minute <= mins(12):
        return get_ratio(minute, mins(5), mins(12))
    elif mins(15) <= minute <= mins(18):
        return 1.0 - get_ratio(minute, mins(15), mins(18))
    return 1.0

temp_i = 0
temps = [15, 15, 30, 30]
while is_running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False
    window_surface.blit(background, (0, 0))

    minute += 1
    if minute >= minute_mod:
        minute = 0
        workday = not workday
        # factoryState.temp = random.randint(0, 40)
        temp_i = (temp_i + 1) % 4
        factoryState.temp = temps[temp_i]

    factoryState.employees_exist = workday and (mins(9) <= minute <= mins(19))
    factoryState.brightness = get_brightness(minute)

    r, g, b = 135, 206, 255
    br = factoryState.brightness
    color = pygame.Color(int(r * br), int(g * br), int(b * br))
    pygame.draw.rect(window_surface, color, [0, 0, 1200, 600])

    lights_on =  bool(light_model(list_to_tensor(factoryState.to_list_light())).max(1).indices.view(1,1)[0][0])
    ac_on =  bool(ac_model(list_to_tensor(factoryState.to_list_ac())).max(1).indices.view(1,1)[0][0])

    # Draw the floor (grass).
    pygame.draw.rect(window_surface, pygame.Color(0, 200, 0), [0, 500, 1200, 100])
    # Draw the building
    pygame.draw.rect(window_surface, pygame.Color(200, 0, 0), [400, 100, 700, 400], 20)
    # Draw the lighting if exists
    if lights_on:
        pygame.draw.rect(window_surface, COLOR_YELLOW, [420, 120, 660, 360])

    text_surface = my_font.render(f"TIME: {mins_to_time(minute)}", False, COLOR_RED)
    window_surface.blit(text_surface, (0, 0))
    text_surface = my_font.render("AC: ON" if ac_on else "AC: OFF", False, COLOR_RED)
    window_surface.blit(text_surface, (0, 50))
    text_surface = my_font.render("LIGHTS: ON" if lights_on else "LIGHTS: OFF", False, COLOR_RED)
    window_surface.blit(text_surface, (0, 100))
    text_surface = my_font.render(f"TEMP: {factoryState.temp}°C" , False, COLOR_RED)
    window_surface.blit(text_surface, (0, 150))
    text_surface = my_font.render("WORKDAY: YES" if workday else "WORKDAY: No", False, COLOR_RED)
    window_surface.blit(text_surface, (0, 200))

    if factoryState.employees_exist:
        window_surface.blit(worker_sprite, (700, 250))

    pygame.display.update()

    pygame.time.wait(5)


pygame.quit()