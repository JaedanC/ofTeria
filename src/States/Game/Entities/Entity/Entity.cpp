#include "Entity.h"
#include "../EntityController.h"

Entity::Entity(EntityController* entityController)
	: entityController(entityController->weak_from_this())
{
	cout << "Making Entity\n";
}

inline ofVec2f* Entity::getWorldPos()
{
	return nullptr;
}
