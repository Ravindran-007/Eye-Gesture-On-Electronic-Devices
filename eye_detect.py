import pyttsx3

engine = pyttsx3.init()

class MyClass:
    def __init__(self):
        pass

    def Blink(self, code):
        if code == 1:
            engine.say("Light on")
            engine.runAndWait()
            print("Light on")

        elif code == 2:
            engine.say("Light off")
            engine.runAndWait()
            print("Light off")

        elif code == 3:
            engine.say("Fan on")
            engine.runAndWait()
            print("Fan on")

        elif code == 4:
            engine.say("Fan off")
            engine.runAndWait()
            print("Fan off")

        elif code == 5:
            engine.say("Bed lamp on")
            engine.runAndWait()
            print("Bed lamp on")

        elif code == 6:
            engine.say("Bed lamp off")
            engine.runAndWait()
            print("Bed lamp off")
