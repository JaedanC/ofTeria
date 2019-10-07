#include "ConsoleParser.h"
#include "Commands.h"

ConsoleParser ConsoleParser::instance;

ConsoleParser::ConsoleParser()
{
	// Add bind command
	commandMap["bind"] = {
		"bind",
		"bind <key> <alias> binds a key to an alias.\n",
		2,
		bind
	};
}

void ConsoleParser::run(vector<string>& command)
{
	if (command.size() < 1) {
		return;
	}
	
	if (commandMap.count(command[0]) == 0) {
		cout << "No such command '" << command[0] << "' exists.\n";
		return;
	}

	Command& c = commandMap[command[0]];
	if (c.parameters > (int)command.size() - 1 || c.parameters == -1) {
		cout << c.command << ": Incorrect parameter amount. Expected " << c.parameters << ". Provided " << command.size() - 1 << endl;
		cout << "Usage: " << c.commandHelp;
		return;
	}
	command.erase(command.begin());
	c.function(command);
}
