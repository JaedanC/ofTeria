#pragma once
#ifndef KEYBOARD_INPUT_H
#define KEYBOARD_INPUT_H

#include "ofMain.h"
#include "ofxGameStates/ofxGameEngine.h"

/* Classes that wish to be notified about exact key presses (which could be a text field)
should inherit from this Callback interface. Currently you must implement both functions.
To be notified the class should add itself to the Callback list. This is done by calling
KeyboardInput::Instance()->registerKeyPressedCallback(this);
	(and/or)
KeyboardInput::Instance()->registerKeyReleasedCallback(this);
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
*/
class KeyboardCallbacks {
public:
	virtual void keyPressed(int key) = 0;
	virtual void keyReleased(int key) = 0;
};


class KeyboardInput {
	friend ofxGameState;
public:
	/*
	These functions directly wrap around OpenFrameworks existing callback functions in the ofApp.h
	They allow us to redirect the input as required to the desired states or callback functions
	ourselves.
	*/
	void keyPressed(int key);
	void keyReleased(int key);
	void mousePressed(int x, int y, int button);
	void mouseReleased(int x, int y, int button);

	/* Call this function to mark a Class as wanting to recieve direct keyPressed Callbacks
	in their keyPressed(int key) function. This won't add an instance twice and it will
	continue to send callbacks until the deregisterKeyPressed() function has been called. */
	void registerKeyPressedCallback(KeyboardCallbacks * callbackInstance);
	void registerKeyReleasedCallback(KeyboardCallbacks* callbackInstance);

	/* Call this function to tell the KeyboardInput class to no longer send keyPressed callbacks
	to this instance anymore. This won't error if you specify an instance that does not exist. */
	void deregisterKeyPressedCallback(KeyboardCallbacks* callbackInstance);
	void deregisterKeyReleasedCallback(KeyboardCallbacks* callbackInstance);

	/* Call this function to bind an alias to a key. Typical Valve usage would be
	bind a +jump      OR     registerAlias("jump", 'a');
	It is not recommended to query a key directly as this does not allow for keybindings to exist.*/
	void registerAlias(string alias, int key);

	/* Call this function at the end of call loop to reset the keyPressed and keyReleased unordered_maps
	for every each loop. They are reassigned by keyPressed() and keyReleased() whenever they are called. */
	void resetPollingMaps();

	static KeyboardInput* Instance() {
		return &instance;
	}
protected:
	KeyboardInput() {}
private:
	static KeyboardInput instance;

	set<KeyboardCallbacks*> keyPressedCallbacks;
	set<KeyboardCallbacks*> keyReleasedCallbacks;
	unordered_map<int, bool> keyPressedMap;
	unordered_map<int, bool> keyDownMap;
	unordered_map<int, bool> keyReleasedMap;
	unordered_map<string, vector<int>> aliasMappings;

	/* Use these functions to query that an alias has been called globally. Not recommended to use unless
	you have checked the stack to see if this alias is for you to use. ofxGameState::queryAliasPressed(string alias)
	for general use instead to ensure that your State should gain access to this alias after it has been
	passed down the stack. */
	bool queryAliasPressed(string alias);
	bool queryAliasDown(string alias);
	bool queryAliasReleased(string alias);

	/* Use these functions to query that a key has been called globalily. Not recommended for general use.
	use registerAlias(string alias, int key) to bind a key and then use ofxGameState::queryAliasPressed(string alias)
	for general use. */
	bool queryPressed(int key);
	bool queryDown(int key);
	bool queryReleased(int key);
};

#endif /* KEYBOARD_INPUT_H */