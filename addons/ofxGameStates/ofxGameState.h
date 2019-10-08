#pragma once
#ifndef OFX_GAME_STATE_H
#define OFX_GAME_STATE_H

#include "ofMain.h"
#include "ofxGameEngine.h"
#include "../src/Keyboard/KeyboardInput.h"

/* Enum to define Alias' as blocking or passing. Register an alias to PASS
by calling registerAliasBlock(alias, <This Enum>);*/
enum KeyboardInputBlockingType {
	INPUT_BLOCK,
	INPUT_PASS
};

/* Base class for any GameStates. Inherit from this class and implement the abstract functions
to get started. */
class ofxGameState
{
protected:
	/* Should GameStates lower than me in the stack be allowed to draw themselves. */
	bool drawTransparent;

	/* Should GameStates lower than me in the stack be allowed to update themselves. */
	bool updateTransparent;
	string stateName;

	/* Stores the aliasPass data which is passed down through the Stack and updated 
	based on the registeredAliasBlocks of the GameStates higher in the Stack. */
	unordered_map<string, bool> aliasPass;
	unordered_map<int, bool> keyPass;

	/* Stores which Alias' this GameState will block for States lower than it in the
	Stack. Currently, if an alias is not registered as PASSING it is assumed to be BLOCKING. */
	unordered_map<string, KeyboardInputBlockingType> registeredAliasBlocks;
	unordered_map<int, KeyboardInputBlockingType> registeredKeyBlocks;

	/* Any Variables for each State should be added as a protected member and initialised in the
	constructor. */
	ofxGameState(const bool& updateTransparent, const bool& drawTransparent, const string& stateName); 

public:
	inline bool getUpdateTransparent() const { return updateTransparent; }
	inline bool getDrawTransparent() const { return drawTransparent; }
	inline const string& getStateName() const {	return stateName; }
	inline unordered_map<string, bool>* getAliasPasses() { return &aliasPass; }
	inline unordered_map<int, bool>* getKeyPasses() { return &keyPass; }
	inline unordered_map<string, KeyboardInputBlockingType> * getRegisteredAliasBlocks() { return &registeredAliasBlocks; }
	inline unordered_map<int, KeyboardInputBlockingType>* getRegisteredKeyBlocks() { return &registeredKeyBlocks; }

	/* If a State requires an alias to be passed through, it can be registed to be
	passing by using this function. Aliases that are queried but not registered are assumed
	to be BLOCKING, and are not passed onto the next GameState in the stack. */
	void registerAliasBlock(string alias, KeyboardInputBlockingType blockingType);
	void registerKeyBlock(int key, KeyboardInputBlockingType blockingType);

	/* If a GameState wants to query an Alias (Which is an string action bound to a key(s) 
	on the keyboard, this is the function to use. It makes sure the Alias is meant for this
	state, and if not, it will return false. */
	bool queryInput(const string& alias, QueryType queryType = QUERY_PRESSED);

	/* This can be used to query a key that is being pressed. It will only allow be true if
	the key is pressed and the GameState is at the top of the stack. Currently this means that
	keys like CTRL can't be passed through to the next GameState. */
	bool queryInput(const int key, QueryType queryType = QUERY_PRESSED);

	/* All GameStates must implement these functions. */
	virtual void setup() = 0;
	virtual void exit() = 0;
	virtual void update(ofxGameEngine* game) = 0;
	virtual void draw(ofxGameEngine* game) = 0;
	
	/* Unused
	void ChangeState(ofxGameEngine* game, ofxGameState* state) {
		game->ChangeState(state);
	}
	*/
};

#endif /* OFX_GAME_STATE_H */