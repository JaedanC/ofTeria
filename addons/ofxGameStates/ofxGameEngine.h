#pragma once
#ifndef OFX_GAME_ENGINE_H
#define OFX_GAME_ENGINE_H

#include "ofMain.h"
#include "../src/Keyboard/KeyboardInput.h"

class KeyboardCallbacks;
class ofxGameState;
class ofxGameEngine
{
private:
	vector<ofxGameState*> states;
	KeyboardInput keyboardInput;

protected:
	ofxGameEngine() {}

public:
	inline KeyboardInput* getKeyboardInput() { return &keyboardInput; }

	// Returns the size of the GameState stack.
	unsigned int StackSize() const;

	// Returns the GameState on the top of the stack.
	ofxGameState* getState() const;

	// Switch the GameState on the top of the stack to something else.
	void ChangeState(ofxGameState* state);

	// Push a GameState onto the stack
	void PushState(ofxGameState* state);

	// Pop the GameState at the top of the stack
	void PopState();

	// These wrap around ofApp.h functions.
	void setup();
	void update();
	void draw();

	inline static ofxGameEngine* Instance() {
		static ofxGameEngine instance;
		return &instance;
	}
};
#endif /* OFX_GAME_ENGINE_H */