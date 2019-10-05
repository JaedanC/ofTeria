#include "KeyboardInput.h"

KeyboardInput KeyboardInput::instance;

void KeyboardInput::keyPressed(int key)
{
	keyDownMap[key] = true;
	keyPressedMap[key] = true;

	cout << key << endl;
}

void KeyboardInput::keyReleased(int key)
{
	keyDownMap[key] = false;
	keyReleasedMap[key] = true;
}

void KeyboardInput::mousePressed(int x, int y, int button)
{
}

void KeyboardInput::mouseReleased(int x, int y, int button)
{
}

void KeyboardInput::registerAlias(string alias, int key)
{
	aliasMappings[alias].push_back(key);
}

bool KeyboardInput::queryAliasPressed(string alias)
{
	if (aliasMappings.count(alias) == 0) return false;
	for (int& key : aliasMappings[alias]) {
		if (queryPressed(key)) return true;
	}
	return false;
}

bool KeyboardInput::queryAliasReleased(string alias)
{
	if (aliasMappings.count(alias) == 0) return false;
	for (int& key : aliasMappings[alias]) {
		if (queryReleased(key)) return true;
	}
	return false;
}

bool KeyboardInput::queryAliasDown(string alias)
{
	if (aliasMappings.count(alias) == 0) return false;
	for (int& key : aliasMappings[alias]) {
		if (queryDown(key)) return true;
	}
	return false;
}

bool KeyboardInput::queryPressed(int key)
{
	return (keyPressedMap.count(key) == 0) ? false : keyPressedMap[key];
}

bool KeyboardInput::queryReleased(int key)
{
	return (keyReleasedMap.count(key) == 0) ? false : keyReleasedMap[key];
}

bool KeyboardInput::queryDown(int key)
{
	return (keyDownMap.count(key) == 0) ? false : keyDownMap[key];
}

void KeyboardInput::resetPollingMaps()
{
	keyPressedMap.clear();
	keyReleasedMap.clear();
}
