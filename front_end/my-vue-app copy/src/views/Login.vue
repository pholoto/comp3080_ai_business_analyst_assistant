<script setup>
import { ref } from "vue";
import { useRoute, useRouter } from "vue-router";

const email = ref("");
const password = ref("");
const route = useRoute();
const router = useRouter();

const onLogin = () => {
  if (!email.value || !password.value) return alert("Enter email & password");
  // demo auth; replace with real API later
  localStorage.setItem("token", "demo-token");
  router.push(route.query.redirect || "/dashboard");
};
</script>

<template>
  <div style="width:360px; margin:auto; border:1px solid #eee; padding:1rem; border-radius:8px;">
    <h2>Login</h2>
    <input v-model="email" placeholder="Email" style="width:100%; margin:.5rem 0;" />
    <input v-model="password" type="password" placeholder="Password" style="width:100%; margin:.5rem 0;" />
    <button @click="onLogin" style="width:100%;">Sign in</button>
  </div>
</template>
