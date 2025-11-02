import { createRouter, createWebHistory } from "vue-router";

import AppLayout from "../layouts/AppLayout.vue";
import AuthLayout from "../layouts/AuthLayout.vue";
import Login from "../views/Login.vue";
import Dashboard from "../views/Dashboard.vue";
import Ideation from "../views/Ideation.vue";
import Deliverables from "../views/Deliverables.vue";
import NotFound from "../views/NotFound.vue";

const routes = [
  {
    path: "/",
    component: AppLayout,
    meta: { requiresAuth: true }, // 👈 protect this area
    children: [
      { path: "", redirect: "/dashboard" },
      { path: "dashboard", component: Dashboard },
      { path: "ideation", component: Ideation },
      { path: "deliverables", component: Deliverables },
    ],
  },
  {
    path: "/auth",
    component: AuthLayout,
    children: [{ path: "login", component: Login }],
  },
  { path: "/:pathMatch(.*)*", component: NotFound },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

router.beforeEach((to, _from, next) => {
  const isAuthed = !!localStorage.getItem("token");
  if (to.meta.requiresAuth && !isAuthed) {
    return next({ path: "/auth/login", query: { redirect: to.fullPath } });
  }
  if (to.path === "/auth/login" && isAuthed) {
    return next("/dashboard");
  }
  next();
});

export default router;
