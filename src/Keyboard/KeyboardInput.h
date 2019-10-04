#pragma once
#ifndef KEYBOARD_INPUT_H
#define KEYBOARD_INPUT_H

#include "ofMain.h"

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
class KeyboardInput {
public:
	void keyPressed(int key);
	void keyReleased(int key);
	void mousePressed(int x, int y, int button);
	void mouseReleased(int x, int y, int button);

	static KeyboardInput* Instance() {
		return &instance;
	}
protected:
	KeyboardInput() {}
private:
	static KeyboardInput instance;
};

#endif /* KEYBOARD_INPUT_H */