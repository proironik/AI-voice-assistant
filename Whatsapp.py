import webbrowser as wb
import time
import keyboard
import pywhatkit
contacts = {"sahaj": 9123712762,
"ronak": "8100367161",
"arka": "9073218594",
"ansh": "9830322674",
"soumava": "6291696763",
"ayushman": "9903439687",
"shourya": "9831082813",
"nilabh": "7980586479",
"archit": "9330350323",
"krishna": "9007202508",
"shubhayu": "7980423631"
}

num=0
openchat=""
t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
hours = int(time.strftime("%H", t))
mins = int(time.strftime("%M", t)) + 2
def msg(query):
    if "sahaj" in query or "degenerate" in query:
        num = "+91" + contacts.get("sahaj")
        query = query.replace("sahaj","")
        pywhatkit.sendwhatmsg(num, query, hours, mins)
    elif "ronak" in query or "crisis" in query:
        num = "+91" + contacts.get("ronak")
        query = query.replace("ronak","")
        pywhatkit.sendwhatmsg(num, query, hours, mins)
    elif "arka" in query or "god" in query:
        num = "+91" + contacts.get("arka")
        query = query.replace("arka","")
        pywhatkit.sendwhatmsg(num, query, hours, mins)
    elif "ansh" in query or "blackhole" in query:
        num = "+91" + contacts.get("ansh")
        query = query.replace("ansh","")
        pywhatkit.sendwhatmsg(num, query, hours, mins)
    elif "soumava" in query or "momo" in query:
        num = "+91" + contacts.get("soumava")
        query = query.replace("soumava","")
        pywhatkit.sendwhatmsg(num, query, hours, mins)
    elif "ayushman" in query or "perfect" in query:
        num = "+91" + contacts.get("ayushman")
        query = query.replace("ayushman","")
        pywhatkit.sendwhatmsg(num, query, hours, mins)
    elif "shourya" in query or "brother" in query:
        num = "+91" + contacts.get("shourya")
        query = query.replace("shourya","")
        pywhatkit.sendwhatmsg(num, query, hours, mins)
    elif "nilabh" in query or "gamer" in query:
        num = "+91" + contacts.get("nilabh")
        query = query.replace("nilabh", "")
        pywhatkit.sendwhatmsg(num, query, hours, mins)
    elif "archit" in query or "kid" in query:
        num = "+91" + contacts.get("archit")
        query = query.replace("archit","")
        pywhatkit.sendwhatmsg(num, query, hours, mins)
    elif "krishna" in query or "topper" in query:
        num = "+91" + contacts.get("krishna")
        query = query.replace("krishna","")
        pywhatkit.sendwhatmsg(num, query, hours, mins)
        pywhatkit.sendwhatmsg(num, query, hours, mins)
    elif "shubhayu" in query or "giga" in query:
        num = "+91" + contacts.get("shubhayu")
        query = query.replace("shubhayu","")
        pywhatkit.sendwhatmsg(num, query, hours, mins)

