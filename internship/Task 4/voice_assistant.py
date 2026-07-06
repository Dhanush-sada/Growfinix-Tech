import datetime
import webbrowser
import sys

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class VoiceAssistant:
    def __init__(self, name="Assistant"):
        self.name = name
        self.engine = None

        if pyttsx3:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 175)
                self.engine.setProperty("volume", 1.0)
            except Exception as e:
                print(f"[Warning] Could not initialize text-to-speech engine: {e}")
                self.engine = None

        self.recognizer = sr.Recognizer() if sr else None

    def speak(self, text):
        print(f"{self.name}: {text}")
        if self.engine:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"[Warning] Speech output failed: {e}")

    def listen(self):
        if self.recognizer:
            try:
                with sr.Microphone() as source:
                    print("\nListening... (speak now)")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(
                        source, timeout=20, phrase_time_limit=10
                    )

                print("Recognizing...")
                command = self.recognizer.recognize_google(audio)
                print(f"You said: {command}")
                return command.lower()

            except sr.WaitTimeoutError:
                print("[Info] No speech detected in time.")
            except sr.UnknownValueError:
                print("[Info] Could not understand the audio.")
            except sr.RequestError as e:
                print(f"[Info] Speech recognition service error: {e}")
            except OSError as e:
                print(f"[Info] No microphone available: {e}")
            except Exception as e:
                print(f"[Info] Unexpected error during listening: {e}")

        return self.text_fallback()

    def text_fallback(self):
        try:
            command = input("Voice unavailable — please type your command: ")
            return command.lower().strip()
        except (EOFError, KeyboardInterrupt):
            return "exit"

    def process_command(self, command):
        if not command:
            self.speak("Sorry, I didn't catch that. Could you repeat it?")

        elif "hello" in command or "hi" in command:
            self.speak(
                f"Hello! I am {self.name}, your voice assistant. How can I help you?"
            )

        elif "time" in command:
            now = datetime.datetime.now().strftime("%I:%M %p")
            self.speak(f"The current time is {now}.")

        elif "date" in command:
            today = datetime.datetime.now().strftime("%A, %B %d, %Y")
            self.speak(f"Today's date is {today}.")

        elif "open google" in command:
            self.speak("Opening Google.")
            webbrowser.open("https://www.google.com")

        elif "open youtube" in command:
            self.speak("Opening YouTube.")
            webbrowser.open("https://www.youtube.com")

        elif "open" in command and "website" not in command:
            site = command.replace("open", "").strip()
            if site:
                self.speak(f"Opening {site}.")
                webbrowser.open(f"https://www.{site}.com")
            else:
                self.speak("Please tell me which website to open.")

        elif "your name" in command:
            self.speak(f"My name is {self.name}. I'm a rule-based voice assistant.")

        elif "how are you" in command:
            self.speak("I'm just a program, but I'm running smoothly. Thanks for asking!")

        elif "thank" in command:
            self.speak("You're welcome! Happy to help.")

        elif "help" in command:
            self.speak(
                "You can ask me for the time, the date, to open Google or YouTube, "
                "greet me, or say exit to quit."
            )

        elif (
            "exit" in command
            or "quit" in command
            or "stop" in command
            or "bye" in command
        ):
            self.speak("Goodbye! Have a great day.")
            return False

        else:
            self.speak(
                "Sorry, I don't understand that command yet. Say 'help' to see what I can do."
            )

        return True

    def run(self):
        self.speak(
            f"{self.name} is online. Say 'help' to hear what I can do, or 'exit' to quit."
        )
        running = True
        while running:
            command = self.listen()
            running = self.process_command(command)


if __name__ == "__main__":
    assistant = VoiceAssistant(name="Growfinix Assistant")
    try:
        assistant.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(0)
     