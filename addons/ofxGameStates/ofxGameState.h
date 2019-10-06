#pragma once
#ifndef OFX_GAME_STATE_H
#define OFX_GAME_STATE_H

#include "ofMain.h"
#include "ofxGameEngine.h"

/* Enum to define Alias' as blocking or passing. Register an alias to PASS
by calling registerAliasBlock(alias, <This Enum>);*/
enum KeyboardInputBlockingType {
	INPUT_BLOCK,
	INPUT_PASS
};

class ofxGameState
{
protected:
	// Should GameStates lower than me in the stack be allowed to draw themselves.
	bool drawTransparent;
	// Should GameStates lower than me in the stack be allowed to update themselves.
	bool updateTransparent;
	string stateName;

	/* Stores the aliasPass data which is passed down through the Stack and updated 
	based on the registeredAliasBlocks of the GameStates higher in the Stack. */
	unordered_map<string, bool> aliasPass;
	/* Stores which Alias' this GameState will block for States lower than it in the
	Stack. Currently, if an alias is not registered as PASSING it is assumed to be BLOCKING. */
	unordered_map<string, KeyboardInputBlockingType> registeredAliasBlocks;

	/* Any Variables for each State should be added as a protected member and initialised in the
	constructor. */
	ofxGameState(const bool& updateTransparent, const bool& drawTransparent, const string& stateName); 

public:
	inline bool getUpdateTransparent() const { return updateTransparent; }
	inline bool getDrawTransparent() const { return drawTransparent; }
	inline const string& getStateName() const {	return stateName; }
	inline unordered_map<string, bool> * getAliasPasses() { return &aliasPass; }
	inline unordered_map<string, KeyboardInputBlockingType> * getRegisteredAliasBlocks() { return &registeredAliasBlocks; }

	/* If a State requires an alias to be passed through, it can be registed to be
	passing by using this function. Aliases that are queried but not registered are assumed
	to be BLOCKING, and are not passed onto the next GameState in the stack. */
	void registerAliasBlock(string alias, KeyboardInputBlockingType blockingType);
	/* If a GameState wants to query an Alias (Which is an string action bound to a key(s) 
	on the keyboard, this is the function to use. It makes sure the Alias is meant for this
	state, and if not, it will return false. */
	bool queryAliasPressed(string alias);

	// All GameStates must implement these functions
	virtual void setup() = 0;
	virtual void update(ofxGameEngine* game) = 0;
	virtual void draw(ofxGameEngine* game) = 0;
	virtual void exit() = 0;
	
	/* Unused
	void ChangeState(ofxGameEngine* game, ofxGameState* state) {
		game->ChangeState(state);
	}
	*/
};

#endif /* OFX_GAME_STATE_H */