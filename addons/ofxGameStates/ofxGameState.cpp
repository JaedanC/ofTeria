#include "ofxGameState.h"
#include "../src/Keyboard/KeyboardInput.h"


ofxGameState::ofxGameState(const bool& updateTransparent, const bool& drawTransparent, const string& stateName)
	: updateTransparent(updateTransparent), drawTransparent(drawTransparent), stateName(stateName)
{
}

void ofxGameState::registerAliasBlock(string alias, KeyboardInputBlockingType blockingType)
{
	registeredAliasBlocks[alias] = blockingType;
}

void ofxGameState::registerKeyBlock(int key, KeyboardInputBlockingType blockingType)
{
	registeredKeyBlocks[key] = blockingType;
}

bool ofxGameState::queryInput(const string& alias, QueryType queryType)
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
		return ofxGameEngine::Instance()->getKeyboardInput()->queryInput(alias, queryType);
	}

	return false;
}

bool ofxGameState::queryInput(const int key, QueryType queryType)
{
	/* If you have never used this alias register it as blocking. */
	if (registeredKeyBlocks.count(key) == 0) {
		registerKeyBlock(key, INPUT_BLOCK);
	}

	/*
	Are we at the top of the stack? If we are, let all input
	through even if it's not blocked. (Obviously) The top State
	gets priority access to all keys.

	If no State has registered this key assume the input is blocked and the keys is being passed.
	keyPass should contain all the keys from the higher parts of the stack.
	*/

	if (ofxGameEngine::Instance()->getState() == this ||
		(keyPass.count(key) != 0 && keyPass[key]))
	{
		return ofxGameEngine::Instance()->getKeyboardInput()->queryInput(key, queryType);
	}

	return false;
}
