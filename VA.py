import pyttsx3
import speech_recognition as sr
import time
import datetime
import wikipedia
import webbrowser
import pywhatkit
import pyautogui
import pyjokes
import operator
import os
import keyboard
import speedtest
import wolframalpha as wolframalpha
from PyDictionary import PyDictionary as diction
from song import son
from song import roast
from Whatsapp import msg
import random
from yeet import game

cond=1
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
# print(voices[1].id)
engine.setProperty('voice', voices[1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak("Good Morning!")

    elif hour>=12 and hour<18:
        speak("Good Afternoon!")

    else:
        speak("Good Evening!")

def wolfram(query):
    api_key = "XK3VEX-3VWL7W44AP"
    requester = wolframalpha.Client(api_key)
    requested = requester.query(query)
    try:
        Answer = next(requested.results).text
        return Answer
    except Exception:
        speak("sorry I didnt understand")

def takeCommand():
    #It takes microphone input from the user and returns string output

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")

    except Exception as e:
        # print(e)
        # print("Say that again please...")
        speak("I didn't understand")
        return "None"
    return query

if __name__ == "__main__":
    speak("yo")
    wishMe()
    while True:
    # if 1:
        query = takeCommand().lower()
        if cond==1:
            # System section
            if "mute" in query:
                speak("got you bro")
                pyautogui.press("volumemute")
            elif "volume up" in query:
                speak("got you bro")
                for i in range(5):
                    pyautogui.press("volumeup")
            elif "volume down" in query:
                speak("got you bro")
                for i in range(5):
                    pyautogui.press("volumedown")
            elif "scroll down" in query:
                s = query.split()
                for i in range(int(s[len(s) - 1])):
                    keyboard.press("page_down")
            elif "scroll up" in query:
                s = query.split()
                for i in range(int(s[len(s) - 1])):
                    keyboard.press("page_up")
            elif "type" in query or "send" in query:
                if "type" in query:
                    query = query.replace("type", "")
                    keyboard.write(query)
                else:
                    query = query.replace("send", "")
                    keyboard.write(query)
                    keyboard.press_and_release("enter")
            elif "speed test" in query:
                speak("got you bro")
                print("running test...")
                s = speedtest.Speedtest()
                s.get_best_server()
                s.download()
                s.upload()
                res = s.results.dict()
                correctDown = int(res["download"]/800000)
                correctUpload = int(res["upload"]/800000)
                print(f'''
                upload: {correctUpload}
                download: {correctDown}
                ping: {res["ping"]}''')
                speak(f"The Downloading is {correctDown} and The Uploading Speed is {correctUpload} mp")
            elif "reader mode" in query:
                speak("got you bro")
                while True:
                    keyboard.press_and_release("page_down")
                    time.sleep(30)
            elif "that thing" in query:
                pywhatkit.playonyt("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
                exit()

                # About section
            elif "who are you" in query:
                speak("I am Friday, ur personal sussy baka")
            elif "what version" in query:
                speak("its beta v04")
            elif "what can you do" in query or "what are the things that you can do" in query or "what else can you do" in query:
                speak("this is what I can do")
                print('''
                .open apps
                .type what you speak
                .search queries on chrome and youtube
                .answer queries from wikipedia
                .run an internet speed test
                .crack jokes
                .insult you
                .set a timer
                .tell the time
                .tell the temperature of a place
                .control system volume
                .play music
                .play vids on youtube
                .maths
                .send messages on whatsapp
                .send emails
                .complete chrome voice automation
                .complete youtube voice automation
                .play perfect guess game
                .scroll up and down pages and reader mode
                .tell meanings and synonyms of words
                .manage classrooms
                .do that thing 
                ''')

            # game section
            elif "game" in query or "perfect guess" in query:
                speak("got you bro")
                game()

            # Youtube section
            elif "open youtube" in query:
                speak("got you bro")
                webbrowser.open("https://www.youtube.com/")
            elif "search on youtube" in query:
                query = query.replace("search on youtube", "")
                speak("got you bro")
                webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
            elif "pause" in query or "continue" in query:
                keyboard.press("k")
            elif "decrease" in query and "youtube" in query:
                for i in range(10):
                    keyboard.press("down")
            elif "increase" in query and "youtube" in query:
                for i in range(10):
                    keyboard.press("up")
            elif "skip by" in query:
                try:
                    s = query.split()
                    n = int(s[len(s) - 2])
                    while n > 0:
                        keyboard.press("right")
                        n = n - 5
                        time.sleep(0.5)
                except Exception:
                    print("sorry, I didn't understand")
            elif "go back by" in query:
                try:
                    s = query.split()
                    n = int(s[len(s) - 2])
                    while n > 0:
                        keyboard.press("left")
                        n = n - 5
                        time.sleep(0.5)
                except Exception:
                    print("sorry, I didn't understand")
            elif "skip" in query and ("5" in query or "five" in query):
                keyboard.press("right")
            elif "go back" in query and ("5" in query or "five" in query):
                keyboard.press("left")
            elif "skip" in query and ("10" in query or "ten" in query):
                keyboard.press("l")
            elif "go back" in query and ("10" in query or "ten" in query):
                keyboard.press("j")
            elif "mute" in query and "youtube" in query:
                keyboard.press("m")
            elif "full screen" in query:
                keyboard.press("f")
            elif "play" in query and "youtube" in query:
                try:
                    query = query.split("play", "")
                    query = query.split("youtube", "")
                    query = query.split("on", "")
                except Exception:
                    pass
                pywhatkit.playonyt(query)

            # Google section
            elif "open google" in query:
                speak("got you bro")
                webbrowser.open("https://google.com")
            elif "wikipedia" in query:
                speak("got you bro")
                query = query.replace("wikipedia", "")
                results = wikipedia.summary(query, sentences=2)
                speak("according to wikipedia")
                speak(results)
            elif "search" in query:
                speak("got you bro")
                query = query.replace("search", "")
                pywhatkit.search(query)
            elif "new tab" in query:
                speak("got you bro")
                keyboard.press_and_release("ctrl + n")
            elif 'close tab' in query:
                speak("got you bro")
                keyboard.press_and_release('ctrl + w')
            elif 'new window' in query:
                speak("got you bro")
                keyboard.press_and_release('ctrl + n')
            elif 'history' in query:
                speak("got you bro")
                keyboard.press_and_release('ctrl + h')
            elif 'incognito' in query:
                speak("got you bro")
                keyboard.press_and_release('ctrl + shift + n')
            elif 'download' in query:
                speak("got you bro")
                keyboard.press_and_release('ctrl + j')
            elif "message" in query:
                speak("got you bro")
                query = query.replace("message","")
                msg(query)
            elif "open amazon" in query:
                speak("got you bro")
                webbrowser.open("https://www.amazon.in/")

            # Classroom section
            elif "open" in query and "computer" in query:
                speak("got you bro")
                webbrowser.open("https://classroom.google.com/u/1/c/MzI2NTczNjEwMDQ1")
            elif "open" in query and "maths" in query:
                speak("got you bro")
                webbrowser.open("https://classroom.google.com/u/1/c/MzI2NTU1MzI0ODI4")
            elif "open" in query and "physics" in query:
                speak("got you bro")
                webbrowser.open("https://classroom.google.com/u/1/c/MzMwMzc0NTk1Mjcz")
            elif "open" in query and "english" in query:
                speak("got you bro")
                webbrowser.open("https://classroom.google.com/u/1/c/MzI2NjI4Mzc3NTIx")
            elif "open" in query and "chemistry" in query:
                speak("got you bro")
                webbrowser.open("https://classroom.google.com/u/1/c/MzMwMzY3MzY2Mjk2")

            # Time section
            elif 'the time' in query:
                strTime = datetime.datetime.now().strftime("%H:%M:%S")
                speak(f"got you bro, the time is {strTime}")
            elif "timer" in query:
                speak("got you bro")
                try:
                    s = query.split()
                    n = int(s[len(s) - 2])
                    while n>0:
                        print(n)
                        n = n - 1
                        time.sleep(1)

                except Exception:
                    print("sorry, I didn't understand")

            # music section
            elif "play" in query and ("music" in query or "song" in query):
                speak("what to play")
                musicname = takeCommand().lower()
                print(musicname)
                try:
                    speak("got you bro")
                    music_dir = "D:\\Music\\"
                    music_dir2 = "D:\\Music2\\"
                    songs = os.listdir(music_dir)
                    songs2 = os.listdir(music_dir2)
                    for item in songs:
                        print(item)
                    son(musicname, music_dir, music_dir2, songs, songs2)
                    # speak("ok")
                    exit()
                except Exception:
                    pass
            elif "play" in query:
                query = query.replace("play","")
                print(query)
                try:
                    speak("got you bro")
                    music_dir = "D:\\Music\\"
                    music_dir2 = "D:\\Music2\\"
                    songs = os.listdir(music_dir)
                    songs2 = os.listdir(music_dir2)
                    for item in songs2:
                        print(item)
                    son(query, music_dir, music_dir2, songs, songs2)
                    exit()
                except Exception:
                    pass

            # maths section
            elif "calculate" in query:
                speak("yes bro, what you want ")
                calc=takeCommand().lower()
                print(calc)
                def get_operator_fn(op):
                    return {
                        '+' : operator.add,
                        'plus' : operator.add,
                        '-' : operator.sub,
                        'minus' : operator.sub,
                        'x' : operator.mul,
                        '/' : operator.__truediv__,
                        'divided' : operator.__truediv__,
                    }[op]
                def eval_binary_expr(op1, oper, op2):
                    op1, op2 = int(op1), int(op2)
                    return get_operator_fn(oper)(op1, op2)
                try:
                    speak("its")
                    speak(eval_binary_expr(*(calc.split())))
                    print(eval_binary_expr(*(calc.split())))
                except Exception:
                    speak("sorry, I couldn't understand")

            #Temperature section
            if "temperature" in query:
                s = query.split()
                temp_query = (s[len(s) - 1])
                speak("got you bro")
                if 'outside' in temp_query:
                    var1 = "Temperature in Kolkata"
                    answer = wolfram(var1)
                    speak(f"{var1} Is {answer} .")
                else:
                    var2 = "Temperature in " + temp_query
                    answ = wolfram(var2)
                    speak(f"{var2} Is {answ}")

            # word meaning section
            elif "meaning" in query:
                try:
                    s = query.split()
                    result = diction.meaning(s[len(s)-1])
                    print(result)
                    speak(result)
                except Exception:
                    speak("sorry, didn't get you")
            elif "synonym" in query:
                try:
                    s = query.split()
                    result = diction.synonym(s[len(s) - 1])
                    print(result)
                    speak(result)
                except Exception:
                    speak("sorry, didn't get you")

            # Joke section
            elif "joke" in query:
                joke=pyjokes.get_joke()
                print(joke)
                speak(joke)
            elif "roast me" in query or "insult me" in query:
                rand = random.randint(1, 10)
                insult = roast(rand)
                speak(insult)

        # end programme
        if "terminate" in query or "kill" in query:
            speak("adios")
            exit()

