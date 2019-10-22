#pragma once
#ifndef CONSOLE_PARSER_H
#define CONSOLE_PARSER_H

#include "ofMain.h"
class ConsoleState;

/* Commands in the consoleParser are defined in these structures. */
struct Command {
	string command;
	string commandHelp;
	int parameters; //-1 = inf, 0 = None (god)
	void(*function)(vector<string>);
};

/* This class uses the pystring library to takes in the commands and run the corresponding
Command which is stored in the commands map. */
class ConsoleParser {
private:
	ofColor colorFailed = ofColor(255, 100, 100);

	ConsoleState* consoleState;
	unordered_map<string, Command> commandMap;
public:
	explicit ConsoleParser(ConsoleState* consoleState);

	inline unordered_map<string, Command>& getCommands() { return commandMap; }
	bool run(vector<string> command);
	void addCommand(const string& name, const string& help, int parameters, void(*functionToCall)(vector<string>));
};

#endif /* CONSOLE_PARSER_H */