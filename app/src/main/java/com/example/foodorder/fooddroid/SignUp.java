package com.example.foodorder.fooddroid;

import android.app.ProgressDialog;
import android.content.Intent;
import android.graphics.Typeface;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.example.foodorder.fooddroid.Common.Common;
import com.example.foodorder.fooddroid.Model.User;

public class SignUp extends AppCompatActivity {


    TextView welcomeText;
    EditText username;
    EditText password;
    EditText confpassword;
    ProgressDialog pd;

    Button continueBtn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sign_up);

        Typeface typeface = Typeface.createFromAsset(getAssets(), "fonts/DroidSans.ttf");



        welcomeText=findViewById(R.id.welcomeText);
        welcomeText.setTypeface(typeface);

        username=findViewById(R.id.username);
        username.setTypeface(typeface);

        password=findViewById(R.id.pwdBox);
        password.setTypeface(typeface);

        confpassword=findViewById(R.id.confpassword);
        confpassword.setTypeface(typeface);

        continueBtn=findViewById(R.id.btnContinue);
        continueBtn.setTypeface(typeface);

        pd= new ProgressDialog(SignUp.this);
        pd.setMessage("loading");

        continueBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pd.show();


                if(!password.getText().toString().equalsIgnoreCase(confpassword.getText().toString())){
                    Toast.makeText(SignUp.this, "Passwords do not match",
                            Toast.LENGTH_LONG).show();
                }
                else{
                    User u=new User("Rohit","Test");
                    Intent welcomeIntent=new Intent(SignUp.this,Welcome.class);
                    Common.CurrentUser=u;
                    startActivity(welcomeIntent);
                    finish();
                }

                pd.dismiss();
            }
        });
    }
}
