#include "ConsoleParser.h"
#include "Commands.h"
#include "../ConsoleState.h"

ConsoleParser::ConsoleParser(ConsoleState* consoleState)
	: consoleState(consoleState)
{
	// Add bind command
	commandMap["bind"] = {
		"bind",
		"bind <key> <alias> binds a key to an alias.",
		2,
		bind
	};

	commandMap["clear"] = {
		"clear",
		"clear - clears the screen",
		0,
		clear
	};
}

bool ConsoleParser::run(vector<string>& command)
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
		consoleState->addText(ofToString(c.command) + ": Incorrect parameter amount. Expected " + ofToString(c.parameters) + ". Provided " + ofToString(command.size() - 1) + "\n", colorFailed);
		consoleState->addText("Usage: " + ofToString(c.commandHelp), colorFailed);
		return false;
	}
	command.erase(command.begin());
	c.function(command);

	return true;
}
