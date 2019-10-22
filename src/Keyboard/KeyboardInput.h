#pragma once
#ifndef KEYBOARD_INPUT_H
#define KEYBOARD_INPUT_H

#include "ofMain.h"

/* To store the mouse in the keyPress data we add this constant to stop clashes with how OF
exposes mousePressEvent 'key' value. */
#define MOUSE_OFFSET_CONSTANT 256

/* Classes that wish to be notified about exact key presses (which could be a text field)
should inherit from this Callback interface. Currently you must implement both functions.
To be notified the class should add itself to the Callback list. This is done by calling
GameEngine::Instance()->getKeyboardInput()->registerKeyPressedCallback(this);
	(and/or)
GameEngine::Instance()->getKeyboardInput()->registerKeyReleasedCallback(this);
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

/* Queries for keys are one of these types.
Pressed: Activate on key press. 
Down: Activate when the key is down.
Released: Activate when the key is released.
*/
enum QueryType {
	QUERY_PRESSED,
	QUERY_DOWN,
	QUERY_RELEASED
};

/* Classes can register to recieve raw input callbacks. Not many things will use
callback_released. */
enum CallbackType {
	CALLBACK_PRESSED,
	CALLBACK_RELEASED
};

/* This class handles raw Keyboard Input from OpenFrameworks and exposes high level functions that
let you query based on alias's and various different query types. Classes can also opte*/
class KeyboardInput {
public:
	/* These three are closely linked together. SpecialBindings maps wierd keys like 'Alt' or 'F3'
	to strings so that you can refer to them as strings rather than knowing their exact value that
	OpenFrameworks exposes them as. */
	KeyboardInput();
	unordered_map<string, int> specialBindings;

	/* Takes in a special key string and returns the OpenFrameworks defined number for it. Works
	for mouse input too, however this is displaced by the defined constant above. */
	int KeyboardInput::convertStringToKey(const string& str);

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
	continue to send callbacks until the deregisterCallback() function has been called. 
	Specify what callbacks you wish to recieve by using the enum. */
	void registerKeyCallback(KeyboardCallbacks * callbackInstance, CallbackType callbackType=CALLBACK_PRESSED);

	/* Call this function to tell the KeyboardInput class to no longer send keyPressed callbacks
	to this instance anymore. This won't error if you specify an instance that does not exist. 
	Specify what callbacks you wish to deregister by using the enum. */
	void deregisterKeyCallback(KeyboardCallbacks* callbackInstance, CallbackType callbackType = CALLBACK_PRESSED);

	/* Call this function to bind an alias to a key. Typical Valve usage would be
	bind a +jump      OR     registerAlias("jump", "a");
	It is not recommended to query a key directly as this does not allow for keybindings to exist.
	The offset parameter is if you wish to add an offset to the resulting ascii value. For example
	holding ctrl changes the keyboard output number for some reason, so to get back to the corresponding
	letter you have to offset by -96. */
	void registerAlias(const string& alias, const string& key, int offset=0);

	/* Call this function at the end of call loop to reset the keyPressed and keyReleased unordered_maps
	for every each loop. They are reassigned by keyPressed() and keyReleased() whenever they are called. */
	void resetPollingMaps();

private:
	set<KeyboardCallbacks*> keyPressedCallbacks;
	set<KeyboardCallbacks*> keyReleasedCallbacks;
	unordered_map<int, bool> keyPressedMap;
	unordered_map<int, bool> keyDownMap;
	unordered_map<int, bool> keyReleasedMap;
	unordered_map<string, vector<int>> keyAliasMappings;

public:
	/* Use these functions to query that an alias has been called globally. Not recommended to use unless
	you have checked the stack to see if this alias is for you to use. ofxGameState::queryInput(const string& alias)
	for general use instead to ensure that your State should gain access to this alias after it has been
	passed down the stack. */
	bool queryInput(const string& alias, QueryType queryType = QUERY_PRESSED);

	/* Use this function to query that a key has been called globalily. Not recommended for general use.
	use registerAlias(const string& alias, const string& key) to bind a key and then use ofxGameState::queryInput(const string& alias)
	for general use. */
	bool queryInput(const int key, QueryType queryType = QUERY_PRESSED);
};

#endif /* KEYBOARD_INPUT_H */