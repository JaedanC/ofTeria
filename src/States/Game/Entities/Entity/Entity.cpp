#include "Entity.h"

Entity::Entity(EntityController* entityController)
	: entityController(entityController)
{
}

inline ofVec2f* Entity::getWorldPos()
{
	return &worldPos;
}

inline EntityController* Entity::getEntityController()
{
	return entityController;
}
