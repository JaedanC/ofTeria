#pragma once
#ifndef KEYBOARD_INPUT_H
#define KEYBOARD_INPUT_H

#include "ofMain.h"
#include "ofxGameStates/ofxGameEngine.h"

/*
 ___________________________________________________________
|							|								|
|  unordered_map<int, bool>	|  unordered_map<string, int>	|
|___________________________|_______________________________|
|							|								|
|			a				|	   +moveLeft, -moveLeft		|
|			b				|								|
|			c				|								|
|			d				|								|
|			e				|	+openInvent, -openInvent	|
|			f				|	  +interact, -interact		|
|			g				|	   +grenade, -grenade		|
|			h				|								|
|			i				|								|
|___________________________|_______________________________|

Press	Bindings = unordermap<string, char> "+Prefix"
Down	Bindings = unordermap<string, char> "no Prefix"
Release Bindings = unordermap<string, char> "-Prefix"

keyboard.getBindPressed("+moveleft");
keyboard.getBindDown("moveleft");
keyboard.getBindReleased("-moveleft");

Press Timeline
|+-------|
|+++++++-|
|-------+|

bool getBindPressed(string& binding) {
	int key = pressBindings[binding];
	return pressedKeys[key];
}

bool getBindReleased(string& binding) {
	int key = releaseBindings[binding];
	return releasedKeys[key];
}

bool getBindDown(string& binding) {
	int key = downBindings[binding];
	return downKeys[key];
}
*/

// Interface
class KeyboardCallbacks {
public:
	virtual void keyPressed(int key) = 0;
	virtual void keyReleased(int key) = 0;
};


class KeyboardInput {
public:
	void keyPressed(int key);
	void keyReleased(int key);
	void mousePressed(int x, int y, int button);
	void mouseReleased(int x, int y, int button);

	void registerPressedCallback(KeyboardCallbacks * callbackInstance);
	//void registerDownCallback(KeyboardCallbacks * callbackInstance);
	void registerReleasedCallback(KeyboardCallbacks* callbackInstance);
	void registerAlias(string alias, int key);

	bool queryAliasPressed(string alias);
	bool queryAliasDown(string alias);
	bool queryAliasReleased(string alias);

	void resetPollingMaps();

	static KeyboardInput* Instance() {
		return &instance;
	}
protected:
	KeyboardInput() {}
private:
	static KeyboardInput instance;

	vector<KeyboardCallbacks*> keyPressedCallbacks;
	//vector<KeyboardCallbacks*> keyDownCallbacks;
	vector<KeyboardCallbacks*> keyReleasedCallbacks;
	unordered_map<int, bool> keyPressedMap;
	unordered_map<int, bool> keyDownMap;
	unordered_map<int, bool> keyReleasedMap;
	unordered_map<string, vector<int>> aliasMappings;

	bool queryPressed(int key);
	bool queryReleased(int key);
	bool queryDown(int key);

	friend class ofxGameEngine;
};

#endif /* KEYBOARD_INPUT_H */