from game_client import RoomBot

def controller(obs):
    nav = obs["navigation"]

    steering = nav["heading_error"] * 0.5
    throttle = 0.7

    return throttle, steering


bot = RoomBot(
    "https://ml.ferit.tech",
    room="testroom",
    name="iramo23"
)

bot.run(controller, hz=20.0)