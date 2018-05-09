package com.example.foodorder.fooddroid;


import android.content.Intent;
import android.graphics.Typeface;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;


public class MainActivity extends AppCompatActivity {


    LinearLayout centerLayout;
    LinearLayout loginLayout;
    LinearLayout termsandService;
    Button createAccount;
    Animation ani;
    TextView txtTerms;
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        centerLayout= findViewById(R.id.centerLogolayout);
        loginLayout=findViewById(R.id.loginLayout);
        termsandService=findViewById(R.id.termsandservice);
        createAccount=findViewById(R.id.btnCreateAccount);
        txtTerms=findViewById(R.id.termstext);

        Typeface typeface = Typeface.createFromAsset(getAssets(), "fonts/DroidSans.ttf");
        createAccount.setTypeface(typeface);
        txtTerms.setTypeface(typeface);

        createAccount.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent signUp=new Intent(MainActivity.this,SignUp.class);
                startActivity(signUp);
            }
        });
    }

    @Override
    protected void onPostCreate(@Nullable Bundle savedInstanceState) {


        super.onPostCreate(savedInstanceState);


        ani= AnimationUtils.loadAnimation(MainActivity.this, R.anim.anim);
        ani.setDuration(1000);
        centerLayout.setAnimation(ani);
        centerLayout.animate();
        ani.start();
        ani.setAnimationListener(new Animation.AnimationListener() {
            @Override
            public void onAnimationStart(Animation animation) {

            }

            @Override
            public void onAnimationEnd(Animation animation) {
                loginLayout.setVisibility(View.VISIBLE);
                termsandService.setVisibility(View.VISIBLE);
            }

            @Override
            public void onAnimationRepeat(Animation animation) {

            }
        });




    }

}
