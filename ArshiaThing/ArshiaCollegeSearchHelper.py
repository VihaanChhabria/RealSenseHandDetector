import keyboard
import webbrowser
import time
import configparser
from sys import exit as sys_exit

global config
config = configparser.ConfigParser()
config.read('ArshiaThing\InfoConfig.ini')
print(config["PAST"]["last_section"])

info = [
    "Undergraduate Enrollment", 
    "Yearly Avg Cost", 
    "Acceptance Rate", 
    "Early Decision Start", 
    "Early Decision Deadline", 
    "Early Action Start", 
    "Early Action Deadline", 
    "Regular Start", 
    "Regular Deadline", 
    "SAT Deadline", 
    "FAFSA Deadline", 
    "Transcript Required", 
    "SAT/ACT Required",
    "Application Fee"
]

colleges = [
    "University of Pennsylvania",
    "Johns Hopkins University",
    "Georgia Institute of Technology",
    "Villanova University",
    "Carnegie Mellon University",
    "New York University",
    "Case Western Reserve University",
    "Boston University",
    "University of Maryland",
    "Virginia Tech",
    "Bucknell",
    "Ohio State University",
    "University of Pittsburgh",
    "Purdue University",
    "Pennsylvania State University",
    "Rutgers University",
    "Rochester Institute of Technology",
    "University of Delaware",
    "Drexel University",
    "Temple University"
]

def remove_elements_before(lst, element):
    if element in lst:
        index = lst.index(element)
        return lst[index:]
    else:
        return lst

def wait_for_button_press(college, section):
    print("Press the button...")

    while True:
        if keyboard.is_pressed('~'):  # Replace 'enter' with the desired button key or combination
            print("Button pressed!")
            time.sleep(0.5)
            break

        if keyboard.is_pressed('esc'):
            time.sleep(0.5)
            config['PAST']['last_section'] = info(info.index(section)-1)
            config['PAST']['last_college'] = colleges(colleges.index(college)-1)
            with open('ArshiaThing\InfoConfig.ini', 'w') as configfile:
                config.write(configfile)
            sys_exit("Finished")

if not config["PAST"]["last_section"] == 'None':
    colleges = remove_elements_before(colleges, config["PAST"]["last_college"])
    info = remove_elements_before(info, config["PAST"]["last_section"])

for college in colleges:
    for section in info:
        wait_for_button_press(college, section)

        section = section.replace(' ', '+')
        college = college.replace(' ', '+')
        query = f"{section}+{college}+Engineering"
        webbrowser.open(f"https://www.google.com/search?q={query}")