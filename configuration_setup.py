import configparser
import argparse

parser = argparse.ArgumentParser(description="Create configuration file by using --conf flag and providing the pat to .ini file as an argument")
parser.add_argument("--conf", help="provide path for configuration file")

#get the path to configuration file
args = parser.parse_args()
configuration_file = args.conf

#read the configuration file 
config = configparser.ConfigParser()
config.read(configuration_file)
config_content = {}

#save the paths and everything important into a python file, so i can import it as a module
for section in config.sections():
	config_content[section] = dict(config[section])

#save in a python file so can be later imported as a module 
with open("config.py", 'w') as f:
	f.write("config="+str(config_content))