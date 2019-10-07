#pragma once
#ifndef CONSOLE_PARSER_H
#define CONSOLE_PARSER_H

#include "ofMain.h"

struct Command {
	string command;
	string commandHelp;
	int parameters; //-1 = inf, 0 = None (god)
	void(*function)(vector<string>);
};

class ConsoleParser {
private:
	unordered_map<string, Command> commandMap;

	static ConsoleParser instance;
protected:
	ConsoleParser();
public:
	static ConsoleParser* Instance() { return &instance; }

	void run(vector<string>& command);
};

#endif /* CONSOLE_PARSER_H */