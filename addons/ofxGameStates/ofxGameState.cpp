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

bool ofxGameState::queryAliasDown(string alias)
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
		return KeyboardInput::Instance()->queryAliasDown(alias);
	}

	return false;
}

bool ofxGameState::queryAliasReleased(string alias)
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
		return KeyboardInput::Instance()->queryAliasReleased(alias);
	}

	return false;
}

bool ofxGameState::queryPressed(int key)
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
		return KeyboardInput::Instance()->queryPressed(key);
	}

	return false;
}

bool ofxGameState::queryDown(int key)
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
		return KeyboardInput::Instance()->queryDown(key);
	}

	return false;
}

bool ofxGameState::queryReleased(int key)
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
		return KeyboardInput::Instance()->queryReleased(key);
	}

	return false;
}

void ofxGameState::registerAliasBlock(string alias, KeyboardInputBlockingType blockingType)
{
	registeredAliasBlocks[alias] = blockingType;
}

void ofxGameState::registerKeyBlock(int key, KeyboardInputBlockingType blockingType)
{
	registeredKeyBlocks[key] = blockingType;
}
