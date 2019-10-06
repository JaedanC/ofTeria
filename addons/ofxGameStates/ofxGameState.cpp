#include "ofxGameState.h"
#include "../src/Keyboard/KeyboardInput.h"


ofxGameState::ofxGameState(const bool& updateTransparent, const bool& drawTransparent, const string& stateName)
	: updateTransparent(updateTransparent), drawTransparent(drawTransparent), stateName(stateName)
{
}


bool ofxGameState::queryAliasPressed(string alias)
{
	/* If you have never used this alias register it as blocking. */
	if (registeredAliasBlocks.count(alias) == 0) {
		registerAliasBlock(alias, INPUT_BLOCK);
	}

	/*
	Are we at the top of the stack? If we are, let all input
	through even if it's not blocked. (Obviously) The top State
	gets priority access to all Aliases. 

	If no State has registered this alias assume the input is blocked and the alias is being passed.
	aliasPass should contain all the keys from the higher parts of the stack.
	*/

	if (ofxGameEngine::Instance()->getState() == this ||
	   (aliasPass.count(alias) != 0 && aliasPass[alias]))
	{
		return KeyboardInput::Instance()->queryAliasPressed(alias);
	}

	return false;
}

void ofxGameState::registerAliasBlock(string alias, KeyboardInputBlockingType blockingType)
{
	registeredAliasBlocks[alias] = blockingType;
}
