#include "ConsoleParser.h"
#include "Commands.h"
#include "../ConsoleState.h"

ConsoleParser::ConsoleParser(ConsoleState* consoleState)
	: consoleState(consoleState)
{
	// Add Commands
	addCommand("bind", "bind <key> <alias> - Binds a key to an alias", 2, bind);
	addCommand("clear", "clear - (the console)", 0, clear);
	addCommand("help", "help - (show this info)", 0, help);
}

bool ConsoleParser::run(vector<string> command)
{
	if (command.size() < 1) {
		return false;
	}
	
	if (commandMap.count(command[0]) == 0) {
		consoleState->addText("No such command '" + ofToString(command[0]) + "' exists.\n", colorFailed);
		return false;
	}

	Command& c = commandMap[command[0]];
	if (c.parameters > (int)command.size() - 1 || c.parameters == -1) {
		consoleState->addText("Usage: " + ofToString(c.commandHelp), colorFailed);
		return false;
	}

	command.erase(command.begin());
	c.function(command);

	return true;
}

void ConsoleParser::addCommand(const string& name, const string& help, int parameters, void(*functionToCall)(vector<string>))
{
	commandMap[name] = { name, help, parameters, functionToCall };
}
